import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from time import perf_counter
import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.models import model_dict
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import get_context, reset_context, set_context
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
        self.chunk_size = config.chunk_prefill_size

        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = model_dict[hf_config.model_type](hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
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

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # 清理缓存，避免上一个实例的缓存影响可用显存估算
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # 重新获取显存状态（清理后）
        free, total = torch.cuda.mem_get_info()
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 计算每个KV块所需的字节数，并根据剩余显存计算可分配的KV块数量
        # 一个block的大小计算公式： 
        # block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype_size
        #           K、V各占一个     层数     一个block的token数    KV头数      每个头的维度    每参数字节数
        # 2*层数*一个block的token数：代表一个block一共存储多少个KV向量
        # kv头数*每头维度*每参数字节数：代表一个KV向量占用多少字节
        # 每头维度，如果hf_config有head_dim属性就用它，否则用hidden_size/num_attention_heads计算
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # 使用更准确的可用显存估算：可分配显存 = 总显存 * 利用率 - 当前已分配
        available_memory = int(total * config.gpu_memory_utilization) - current
        config.num_kvcache_blocks = available_memory // block_bytes
        
        # 记录KV缓存预算，便于排查因块数不足导致的非法访存
        total_gb = total / (1024 ** 3)
        avail_mb = available_memory / (1024 ** 2)
        block_mb = block_bytes / (1024 ** 2)
        total_token_capacity = config.num_kvcache_blocks * self.block_size
        
        assert config.num_kvcache_blocks > 0
        if config.num_kvcache_blocks <= 0:
            # 回退为至少1块并给出提示，避免直接断言导致无法实例化
            config.num_kvcache_blocks = 1
            print("[WARN] 可用显存不足按估算无法分配KV缓存，已回退为1个KV块。建议调小 max_model_len / chunk_prefill_size 或降低gpu_memory_utilization。")

        # 创建KV缓存张量（在 GPU 上），并分配给模型的每一层
        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
            self.block_size, num_kv_heads, head_dim,
            dtype=hf_config.torch_dtype,
            device='cuda' # 显式指定在GPU上创建，防止CUDA非法内存访问
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [0] * (max_len - len(seq.block_table)) for seq in seqs
            # 这里改成了填0️⃣
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

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
    #         input_ids.extend(seq[seq.num_cached_tokens :])
    #         positions.extend(list(range(seq.num_cached_tokens, seqlen)))
    #         seqlen_q = seqlen - seq.num_cached_tokens
    #         seqlen_k = seqlen
    #         cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    #         cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
    #         max_seqlen_q = max(seqlen_q, max_seqlen_q)
    #         max_seqlen_k = max(seqlen_k, max_seqlen_k)
    #         if not seq.block_table:
    #             continue
    #         for i in range(seq.num_cached_blocks, seq.num_blocks):
    #             start = seq.block_table[i] * self.block_size
    #             if i != seq.num_blocks - 1:
    #                 end = start + self.block_size
    #             else:
    #                 end = start + seq.last_block_num_tokens
    #             slot_mapping.extend(list(range(start, end)))
    #     if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
    #         block_tables = self.prepare_block_tables(seqs)
    #     input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
    #         non_blocking=True
    #     )
    #     positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
    #         non_blocking=True
    #     )
    #     cu_seqlens_q = torch.tensor(
    #         cu_seqlens_q, dtype=torch.int32, pin_memory=True
    #     ).cuda(non_blocking=True)
    #     cu_seqlens_k = torch.tensor(
    #         cu_seqlens_k, dtype=torch.int32, pin_memory=True
    #     ).cuda(non_blocking=True)
    #     slot_mapping = torch.tensor(
    #         slot_mapping, dtype=torch.int32, pin_memory=True
    #     ).cuda(non_blocking=True)
    #     set_context(
    #         True,
    #         cu_seqlens_q,
    #         cu_seqlens_k,
    #         max_seqlen_q,
    #         max_seqlen_k,
    #         slot_mapping,
    #         None,
    #         block_tables,
    #     )
    #     return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], is_prefill:bool = False, num_logits:int = None):
        temperatures = []
        # ---------------- 添加chunked prefill逻辑 ------------------
        # 根据实际生成的 logits 数量来准备 temperature
        if num_logits is not None:
            # 使用实际的 logits 数量来决定需要多少个 temperature
            # 这样可以处理 warmup 等场景下序列数量和 logits 数量不匹配的情况
            if is_prefill and len(seqs) > 1 and num_logits < len(seqs):
                # 混合批次：第一个是 prefill，其余是 decode
                # num_logits < len(seqs) 说明第一个序列没有生成 logit
                start_idx = 1
                end_idx = num_logits + 1
            else:
                # 纯 prefill 或纯 decode，取前 num_logits 个序列
                start_idx = 0
                end_idx = num_logits
            
            for seq in seqs[start_idx:end_idx]:
                temperatures.append(seq.temperature)
        else:
            # 旧逻辑，兼容性保留
            is_mixed_batch = False
            if is_prefill and len(seqs) > 1:
                first_seq = seqs[0]
                remaining = len(first_seq) - first_seq.num_cached_tokens - first_seq.prefilled_len
                if remaining > 0:
                    is_mixed_batch = all(
                        len(s) - s.num_cached_tokens - s.prefilled_len <= 0
                        for s in seqs[1:]
                    )
            
            start_idx = 1 if is_mixed_batch else 0
            for seq in seqs[start_idx:]:
                temperatures.append(seq.temperature)
        # ---------------- 添加chunked prefill逻辑 ------------------
        # for seq in seqs:
        #     temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        ctx = get_context()
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    def run(self, seqs: list[Sequence], is_prefill: bool, num_prefill_tokens: int = 0, num_decode_tokens: int = 0) -> list[int]:
        # input_ids, positions = (
        #     self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # )
        t0 = perf_counter()
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs, num_prefill_tokens, num_decode_tokens)
        else:
            input_ids, positions = self.prepare_decode(seqs)
                
        # DEBUG: 打印输入信息
        if self.rank == 0 and is_prefill and len(seqs) > 1:
            # print(f"\n[DEBUG run] Mixed batch: {len(seqs)} seqs")
            for i, seq in enumerate(seqs):
                remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
                # print(f"  Seq[{i}] id={seq.seq_id}: remaining={remaining}, total_tokens={len(seq)}")
            # print(f"  input_ids.shape: {input_ids.shape}")
        t1 = perf_counter()
        # print(f"prepare_prefill/decode time: {(t1 - t0) * 1000:.2f}ms")
        logits = self.run_model(input_ids, positions, is_prefill)
        t2 = perf_counter()
        # print(f"run model time: {(t2 - t1) * 1000:.2f}ms")


        if is_prefill and len(seqs) > 1:
            # CHUNK_SIZE = 512
            CHUNK_SIZE = self.chunk_size
            # 检查第一个序列是否有 prefill chunk
            prefill_seq = seqs[0]
            prompt_remaining = prefill_seq.num_prompt_tokens - prefill_seq.num_cached_tokens - prefill_seq.prefilled_len
            # print(f"[DEBUG strip] prompt_remaining={prompt_remaining}")
            if prompt_remaining > 0:
                # 第一个序列是 prefill，丢弃第一行 logits（对应prefill序列）
                # print(f"[DEBUG strip] original logits.shape={logits.shape}, removing first sequence logits")
                logits = logits[1:]  # 移除第一行（prefill序列的logits）
                
        num_logits = logits.size(0) if self.rank == 0 else None
        temperatures = self.prepare_sample(seqs, is_prefill, num_logits) if self.rank == 0 else None
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        t3 = perf_counter()
        # print(f"sample time: {(t3 - t2) * 1000:.2f}ms")
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
    # def prepare_prefill(self, seqs: list[Sequence
    def prepare_prefill(self, seqs: list[Sequence], num_prefill_tokens: int = 0, num_decode_tokens: int = 0):
        '''
        支持混合批次：分别处理 prefill 和 decode 序列
        '''
        # CHUNK_SIZE = 512  # 每次prefill处理的chunk大小
        CHUNK_SIZE = self.chunk_size
        # 记录混合批次信息
        is_mixed_batch = num_prefill_tokens > 0 and num_decode_tokens > 0
        context_lens = []  # 添加 context_lens 用于混合批次
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        # for seq in seqs:
        for i, seq in enumerate(seqs):
            seqlen = len(seq)
            # 存储未缓存的token_ids和对应的位置
            # ---------------- 原始代码 ------------------
            # input_ids.extend(seq[seq.num_cached_tokens:])
            # positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            # q序列的长度是未缓存的token数，为什么能减去缓存了的token数？
            # 因为cached token不需要重新计算attention，只需要计算新生成的token对cached token的attention
            # 而k序列的长度是整个序列长度，因为k需要包含所有token用于后续的attention计算
            # seqlen_q = seqlen - seq.num_cached_tokens
            # seqlen_k = seqlen
            # ---------------- 原始代码 ------------------

            # ---------------- 修改chunked prefill逻辑 ------------------

           
            
            # seqlen_q = min(CHUNK_SIZE, seqlen - start_pos)  # q序列长度为一个chunk大小
            # seqlen_k = end_pos # k序列长度只包含已经prefill的token
            # remaining_prefill = seqlen - seq.num_cached_tokens - seq.prefilled_len
            # 计算剩余需要 prefill 的长度（基于 prompt，不包括已生成的 token）
            prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len


            if prompt_remaining > 0: # 避免warmup时，多个序列进行prefill
                # prefill,有CHUNK_SIZE个token要进
                # 从 seq.prefilled_len 开始添加未缓存的 token
                start_pos = seq.num_cached_tokens + seq.prefilled_len
                end_pos = start_pos + min(CHUNK_SIZE, seqlen - start_pos)
                seqlen_q = min(CHUNK_SIZE, prompt_remaining)
                seqlen_k = end_pos
            else:
                # decode,只有一个token要进
                # 已完成 prefill 的序列：当作 decode 处理（只取最新 token）
                start_pos = seqlen - 1
                end_pos = seqlen
                seqlen_q = 1
                seqlen_k = seqlen
            
            assert seqlen_q > 0
             # 本次要处理的token
            input_ids.extend(seq[start_pos:end_pos])
            positions.extend(list(range(start_pos, end_pos)))
            # ---------------- 修改chunked prefill逻辑 ------------------

            # 更新累计长度，用于构建cu_seqlens张量，确定每个序列在张量拼接后的起始位置
            # 要更新时，那旧的最后一个数据加上当前序列的长度，并append到列表末尾
            # 例如，第一个序列长度为3，第二个序列长度为5，则cu_seqlens为[0, 3, 8]
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # 对于混合批次，记录每个序列的完整长度（用于 decode 部分读取 KV cache）
            context_lens.append(seqlen)
            if not seq.block_table:    # warmup，块表为空是预热阶段，直接跳过
                continue
            # for pos in range(start_pos, end_pos):
            #     block_idx = pos // self.block_size
            #     block_offset = pos % self.block_size
            #     if block_idx < len(seq.block_table):
            #         # 添加token到blcok_table的映射位置
            #         slot = seq.block_table[block_idx] * self.block_size + block_offset
            #         slot_mapping.append(slot)
            # 对于 decode 序列（已完成 prefill），使用最后一个 token 的 slot
            # 对于 prefill 序列，遍历所有新 token 的 slot
            if prompt_remaining <= 0:
                # Decode: 使用最后一个已分配的 slot
                slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
                slot_mapping.append(slot)
            else:
                # Prefill: 为每个新 token 分配 slot
                for pos in range(start_pos, end_pos):
                    block_idx = pos // self.block_size
                    block_offset = pos % self.block_size
                    if block_idx < len(seq.block_table):
                        # 添加token到blcok_table的映射位置
                        slot = seq.block_table[block_idx] * self.block_size + block_offset
                        slot_mapping.append(slot)
            # for i in range(seq.num_cached_blocks, seq.num_blocks):
            #     # 计算每个块的开始和结束位置，用于slot_mapping
            #     start = seq.block_table[i] * self.block_size
            #     if i != seq.num_blocks - 1:
            #         end = start + self.block_size
            #     else:
            #         end = start + seq.last_block_num_tokens 
            #     slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            # 说明存在缓存的token，将块表对齐，可以根据块表快速定位缓存位置
            # 这个对齐和同一批次内不同序列之间填充对齐序列长度是不一样的
            # 没有前缀缓存时，对齐序列长度是为了让所有序列的长度相同，便于批量处理
            # 有前缀缓存时，对齐块表是为了让模型能正确访问缓存的KV块，利于KV复用
            block_tables = self.prepare_block_tables(seqs)
        # 将所有数据转换为张量，并移动到GPU上
        # input_ids：所有未被缓存的token_ids，形状为(total_num_tokens)，一维张量
        # pin_memory=True表示在CPU上分配页锁定内存，利于从CPU到GPU的数据传输效率
        # .cuda(non_blocking=True)表示将张量移动到GPU上，异步传输
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if context_lens else None
        
        # 额外打印当前批次涉及的KV占用，辅助定位是否接近KV块上限
        # all_block_ids = [bid for seq in seqs for bid in seq.block_table]
        # max_block_id = max(all_block_ids) if all_block_ids else -1
        # unique_blocks = len(set(all_block_ids))
        # est_tokens_in_cache = unique_blocks * self.block_size
        # print(
        #     f"[KVCache batch] seqs={len(seqs)}, unique_blocks={unique_blocks}, "
        #     f"max_block_id={max_block_id}, est_cached_tokens≈{est_tokens_in_cache}, "
        #     f"block_capacity={self.config.num_kvcache_blocks * self.block_size}"
        # )
        
        # 如果传入的 num_prefill_tokens 和 num_decode_tokens 都是 0（如 warmup 时），
        # 则根据实际的 cu_seqlens_q 推断真实的 token 数量
        if num_prefill_tokens == 0 and num_decode_tokens == 0:
            total_tokens = cu_seqlens_q[-1].item() if isinstance(cu_seqlens_q, torch.Tensor) else cu_seqlens_q[-1]
            # 纯 prefill 批次：所有 token 都是 prefill
            num_prefill_tokens = total_tokens
            num_decode_tokens = 0
        # 设置全局的上下文，信息包括：是否是prefill阶段、累计的序列长度、最大序列长度、映射表、块表等
        # 用于后续模型的前向计算，特别是处理缓存和attention机制时使用
        # set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
         # 额外传递混合批次信息：num_prefill_tokens, num_decode_tokens, context_lens
        print(f"[DEBUG prepare_prefill] calling set_context with is_prefill=True, num_prefill={num_prefill_tokens}, num_decode={num_decode_tokens}")
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables,
                   num_prefill_tokens=num_prefill_tokens, num_decode_tokens=num_decode_tokens)
        ctx = get_context()
        
        
        return input_ids, positions
