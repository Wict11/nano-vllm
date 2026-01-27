import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        # [ ] chunked prefill 的 chunk 大小参数
        self.chunk_size = config.chunked_prefill_size

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda") # 把参数和缓存都放到 GPU 上
        self.model = Qwen3ForCausalLM(hf_config) # 初始化模型
        load_model(self.model, config.model) # 加载模型权重
        self.sampler = Sampler() # 初始化采样器
        self.warmup_model() # 预热模型，分配内存
        # 分配kv cache
        self.allocate_kv_cache()
        if not self.enforce_eager:
            # 会预先跑一遍模型，捕获cuda graph
            self.capture_cudagraph() # cuda graph优化
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # 子进程循环等待主进程的调用
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # 为了消除第一次推理的不确定性，把所有“第一次才发生的事情”提前做掉，
    def warmup_model(self):
        # 重置torch内存状态
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # sequence采用全0填充max_model_len长度，用于预热模型，尽可能跑满内存
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # true表示只跑预填充阶段，因为prefill 是内存分配 + kernel 最复杂的阶段
        # 函数定义：def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self.run(seqs, True)
        # 最后再清空一次cache
        torch.cuda.empty_cache()

# 分配内存
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # free是剩余内存，total是总内存
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # 进程占用峰值内存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # 进程当前占用内存
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # tp并行，每个cpu只处理一部分qkv，world_size是tp size，总头数除以tp size得到每个gpu的头数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 计算每个头对应的隐藏层维度
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 计算单个block占用的字节数。 2表示 key和value * 总层数 * 块大小 * kv头数 * 头维度 * 每个数据类型字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # 计算可以分出多少个block用于kv cache
        # 可利用内存 = 总内存 * 利用率 - 已用 - 峰值 + 当前占用。除以单个block字节数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 维度定义顺序：2表示key和value，层数，块数，块大小，kv头数，头维度
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 对应赋值
        for module in self.model.modules():
            # 只有attention层才有kv cache这两个属性
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence], num_prefill_tokens: int = 0, num_decode_tokens: int = 0):   
        '''
        支持混合批次：分别处理 prefill 和 decode 序列
        '''
        is_mixed_batch = num_prefill_tokens > 0 and num_decode_tokens > 0
        context_lens = []
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for i, seq in enumerate(seqs):
            seq_len = len(seq)
            # 计算剩余需要prefill的token数量
            prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
            if prompt_tokens_left > 0:
                # --- Chunked Prefill 核心逻辑 ---
                # 起点：接着上次没处理完的地方继续
                # 从已经prefill的token位置开始，添加chunk大小的token
                start_pos = seq.num_cached_tokens + seq.prefilled_tokens
                # 终点：起点 + (Chunk Size 和 剩余长度 取更小值)
                end_pos = start_pos + min(self.chunk_size, seq_len - start_pos)
                # Q 的长度：这次只算这一块
                seqlen_q = min(self.chunk_size, prompt_tokens_left)
                # K 的长度：虽然 Q 只算一块，但 K 需要包含之前所有的历史（Prefix Cache + 之前的 chunks）
                seqlen_k = end_pos
            else:
                # decode
                # 只添加最后一个token
                start_pos = seq_len - 1
                end_pos = seq_len
                seqlen_q = 1
                seqlen_k = seq_len
            assert seqlen_q > 0
            # 本次需要处理的token ids
            input_ids.extend(seq[start_pos:end_pos])
            positions.extend(list(range(start_pos, end_pos)))
            # FlashAttention（特别是 VarLen 版本）需要知道每个序列的 Q 和 K 分别有多长，才能正确计算注意力
            # 【0， 2， 5， 10 。。。】更新累计长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # 对于混合批次，记录序列完整长度
            # 因为decode阶段需要知道完整上下文长度，用于kvcache索引
            # 混合批次才记录context_lens
            # 当前序列在逻辑上的真实总长度
            # PagedAttention Kernel 读取 context_lens[i]，计算模运算，
            # 从而知道最后一个 Block 的有效边界在哪里，防止读取到垃圾数据导致计算错误。
            context_lens.append(seq_len)
            if not seq.block_table:    # warmup
                continue
            if prompt_tokens_left <= 0:
                # decode阶段，添加最后一个block的slot mapping
                slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
                slot_mapping.append(slot)
            else:
                # prefill阶段，添加本次chunk的slot mapping
                for pos in range(start_pos, end_pos):
                    block_idx = pos // self.block_size
                    block_offset = pos % self.block_size
                    if block_idx <= len(seq.block_table):
                        slot = seq.block_table[block_idx] * self.block_size + block_offset
                        slot_mapping.append(slot)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if context_lens else None
        # 额外传递混合批次信息：num_prefill_tokens, num_decode_tokens, context_lens
        # 数据打包和分发机制
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables,
                num_prefill_tokens=num_prefill_tokens, num_decode_tokens=num_decode_tokens)
        return input_ids, positions
         
            
    # def prepare_prefill(self, seqs: list[Sequence]):
    #     input_ids = []
    #     positions = []
    #     cu_seqlens_q = [0]
    #     cu_seqlens_k = [0]
    #     max_seqlen_q = 0
    #     max_seqlen_k = 0
    #     slot_mapping = []
    #     block_tables = None
    #     for seq in seqs:
    #         seqlen = len(seq)
    #         # 横向拼接token ids，展开成一维x列表
    #         # extend是把一个列表的元素逐个添加到另一个列表中
    #         input_ids.extend(seq[seq.num_cached_tokens:])
    #         # 去除掉缓存的部分，只处理新增的token
    #         positions.extend(list(range(seq.num_cached_tokens, seqlen)))
    #         seqlen_q = seqlen - seq.num_cached_tokens
    #         seqlen_k = seqlen
    #         cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    #         cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
    #         max_seqlen_q = max(seqlen_q, max_seqlen_q)
    #         max_seqlen_k = max(seqlen_k, max_seqlen_k)
    #         if not seq.block_table:    # warmup
    #             continue
    #         for i in range(seq.num_cached_blocks, seq.num_blocks):
    #             start = seq.block_table[i] * self.block_size
    #             if i != seq.num_blocks - 1:
    #                 end = start + self.block_size
    #             else:
    #                 end = start + seq.last_block_num_tokens 
    #             # 保存 slot mapping 信息
    #             slot_mapping.extend(list(range(start, end)))
    #     if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
    #         block_tables = self.prepare_block_tables(seqs)
    #     input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    #     positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    #     cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    #     cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    #     slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    #     set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
    #     return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        # 准备flash attention需要的数据
        for seq in seqs:
            # append是添加整个元素作为一个整体
            # 添加最后一个token用于解码
            input_ids.append(seq.last_token)
            # 添加位置编码
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], is_prefill: bool = False, num_logits: int = None):
        temperatures = []
        # prefill和decode混合批次时，
        # 只有第一个序列是prefill，其他都是decode，
        # prefill不需要logits采样
        # 需要根据实际生成的 logits 数量来准备 temperature
        if num_logits is not None:
            # 使用实际的 logits 数量来决定需要多少个 temperature
            # 这样可以处理 warmup 等场景下序列数量和 logits 数量不匹配的情况
            # 如果有prefill序列，并且序列数大于1（则一定是混合批次）
            if is_prefill and len(seqs) > 1 and num_logits < len(seqs):
                # 混合批次prefill，只有第一个序列有prefill token，其他都是decode token
                # num_logits 小于序列数，说明第一个序列没有生成token
                start_idx = 1
                end_idx = num_logits + 1
            else:
                # 虽然也是混合批次，但 num_logits == len(seqs)
                # 这说明 Prefill 任务处理的是最后一个 Chunk
                # Prompt 读完了，现在需要预测由 Prompt 引导出来的第一个新 token
                # 所以 seqs[0] 也需要采样，也需要 Logits
                # 纯prefill批次，或者decode批次，取前num_logits个序列
                start_idx = 0
                end_idx = num_logits
            for seq in seqs[start_idx:end_idx]:
                temperatures.append(seq.temperature)
        else:
            # 兜底逻辑
            # 如果没传入 num_logits，则默认所有序列都需要采样，这时候就要筛选出混合批次的情况
            is_mixed_batch = False
            if is_prefill and len(seqs) > 1:
                first_seq = seqs[0]
                token_left = first_seq.num_prompt_tokens - first_seq.prefilled_tokens - first_seq.num_cached_tokens
                # 说明 Prompt 还没读完，不需要采样，start_idx = 1
                if token_left > 0:
                    is_mixed_batch = all(len(s) - s.num_cached_tokens -s.num_prompt_tokens <= 0 for s in seqs[1:])
            # token left = 0 说明读完了，需要采样，start_idx = 0
            start_idx = 1 if is_mixed_batch else 0
            for seq in seqs[start_idx:]:
                temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool, num_prefill_tokens: int = 0, num_decode_tokens: int = 0) -> list[int]:
        # input_ids是输入的token id序列，positions是对应的位置编码
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs, num_prefill_tokens, num_decode_tokens)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        logits = self.run_model(input_ids, positions, is_prefill)

        if is_prefill and len(seqs) > 1:
            # 计算 Prefill 任务是不是还有剩余没处理完的 prompt
            prefill_token_left = seqs[0].num_prompt_tokens - seqs[0].prefilled_tokens - seqs[0].num_cached_tokens
            if prefill_token_left > 0:
                # 混合批次prefill，只有第一个序列有prefill token，其他都是decode token
                # prefill还没结束，不需要logits采样，直接返回decode部分的logits
                logits = logits[1:]
        num_logits = logits.size(0) if self.rank == 0 else None
        # 不用多进程做采样，只需要主进程采样即可
        # 温度值
        temperatures = self.prepare_sample(seqs, is_prefill, num_logits) if self.rank == 0 else None
        # 根据概率分布采样出下一个token id
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # max batch size, 限制单批次最大序列数不超过512
        max_bs = min(self.config.max_num_seqs, 512)
        # 是单条请求最大tokens长度向上整除block_size，相当于拉满预先模拟计算量
        # config.max_model_len：模型能处理的最大序列长度（比如 4096）
        # self.block_size：每个计算“块”包含多少 token（比如 128）
        # 我们想知道：需要多少块才能覆盖最大序列长度，需要向下取整
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 开辟cuda graph需要的最大空间
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 定义graph维度，每次来了实际的batch size就向上取整取最接近的
        # 密度刚开始是 [1,2,4,8]，后面是每16个数取一个，直到max_bs
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 倒着遍历，先捕获大batch size的graph，再捕获小batch size的graph
        # 因为小batch size的graph可以复用大batch size的graph pool
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # 做图捕获
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存graph需要的变量
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
