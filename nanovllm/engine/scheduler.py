from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# 实现了continuous batching调度算法
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 单次批量推理允许的最大序列数量
        self.max_num_batched_tokens = config.max_num_batched_tokens # 单次批量推理允许的最大 token 数量
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # [ ] chunked prefill 的 chunk 大小参数
        self.chunk_size = config.chunked_prefill_size
        self.enable_chunked_prefill = config.enable_chunked_prefill

    def is_finished(self):
        # 两个队列都没有序列则表示全部完成
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 直接加入等待队列
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # 1. 首先检查当前running队列中是否有prefill序列
        running_has_prefill = False
        # 记录running队列中prefill未完成的序列
        prefill_seq_found = None

        for seq in self.running:
            if num_seqs >= self.max_num_seqs:
                break
            prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
            if prompt_tokens_left <= 0:
                continue
            new_chunk_size = min(self.chunk_size, prompt_tokens_left)
            # 如果加入该序列会超出token数上限，或者无法分配足够的块，则停止调度
            if num_batched_tokens + new_chunk_size > self.max_num_batched_tokens:
                break
            num_seqs += 1

            num_batched_tokens += new_chunk_size
            scheduled_seqs.append(seq)
            running_has_prefill = True
            prefill_seq_found = seq
            # 每次只插入一个prefill序列的一个chunk
            break
        
        # 从 running 队列中临时移除已调度的 prefill 序列，避免在 decode 阶段重复处理
        if prefill_seq_found is not None:
            self.running.remove(prefill_seq_found)
        
        # 2. 如果running中没有需要prefill的队列，就进waiting取
        if not running_has_prefill and self.waiting and num_seqs <= self.max_num_seqs:
            seq = self.waiting[0]

            prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
            new_chunk_size = min(self.chunk_size, prompt_tokens_left)
            # 如果加入该序列会超出token数上限，或者无法分配足够的块，则停止调度
            # 这里的self.block_manager.can_allocate(seq)会判断全量prompt长度，不是当前chunk的长度
            if new_chunk_size > 0 and num_batched_tokens + new_chunk_size <= self.max_num_batched_tokens and self.block_manager.can_allocate(seq):
                # 加入该序列
                num_seqs += 1
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                # 调整当前序列到运行队列
                self.waiting.popleft()
                self.running.append(seq)
                # 预填充阶段加入调度列表
                scheduled_seqs.append(seq)
                num_batched_tokens += new_chunk_size
                running_has_prefill = True

        # # 循环处理调度等待队列中的序列
        # # 保证不超过单次批量推理的序列数和token数上限
        # while self.waiting and num_seqs < self.max_num_seqs:
        #     seq = self.waiting[0]
        #     # 如果加入该序列会超出token数上限，或者无法分配足够的块，则停止调度
        #     if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
        #         break
        #     num_seqs += 1
        #     self.block_manager.allocate(seq)
        #     # 这里天然是为chunked prefill设计的
        #     num_batched_tokens += len(seq) - seq.num_cached_tokens
        #     seq.status = SequenceStatus.RUNNING

        #     self.waiting.popleft()
        #     self.running.append(seq)

        #     scheduled_seqs.append(seq)
        
        # # 如果有预填充的序列则直接返回，说明还需要继续进行预填充
        # if scheduled_seqs:
        #     return scheduled_seqs, True

        # decode
        # 原代码默认所有序列都完成了prefill，但是当前是把chunked prefill和decode混合在一起调度的
        # 原代码如果没有需要prefill的序列，则进行解码阶段的调度
        if self.enable_chunked_prefill or not running_has_prefill:
            while self.running and num_seqs < self.max_num_seqs :
                seq = self.running.popleft()
                # 如果无法为该序列追加块，则开始抢占
                # [ ] 关于抢占的知识点
                while not self.block_manager.can_append(seq):
                    if self.running:
                        # 如果有正在运行的序列，从running队列的右侧pop一个seq并抢占资源
                        self.preempt(self.running.pop())
                    else:
                        # 如果没有正在运行的序列，直接将当前序列抢占
                        self.preempt(seq)
                        break
                else:
                    if seq in scheduled_seqs:
                        continue  # 已经调度过该序列，跳过
                    prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
                    if prompt_tokens_left > 0:
                        # 还有prefill未完成的chunk，跳过decode调度
                        # 放回running队列
                        self.running.appendleft(seq)
                        continue
                    if self.block_manager.can_allocate(seq) and num_batched_tokens + 1 <= self.max_num_batched_tokens:
                        # 可以为该序列追加块，且不超出token数上限
                        # BUG 只有在纯decode批次才提前分配块，混合批次不提前分配
                        if not running_has_prefill:
                            # 判断是否需要追加块
                            self.block_manager.may_append(seq)
                        num_batched_tokens += 1
                        num_seqs += 1
                        scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 再塞回去
        self.running.extendleft(reversed(scheduled_seqs))
        num_prefill_tokens = 0
        num_decode_tokens = 0
        for seq in scheduled_seqs:
            prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
            if prompt_tokens_left > 0:
                new_chunk_size = min(prompt_tokens_left, self.chunk_size)
                num_prefill_tokens += new_chunk_size
            else:
                num_decode_tokens += 1
        return scheduled_seqs, running_has_prefill, num_prefill_tokens, num_decode_tokens

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool = False) -> list[bool]:
        '''
        处理模型输出：
        - 如果is_prefill为True，且序列长度超过1，则第一个是prefill的token_ids，后续是decode的token_ids
        - 如果is_prefill为True且序列长度为1，则全部是prefill的token_ids
        - 如果is_prefill为False，则全部是decode的token_ids
        '''
        # [ ] 根据is_prefill标志处理不同情况
        if is_prefill:
            # 混合prefill和decode批次
            # 因为有prefill序列，同时总序列数又不止1个，所以肯定有decode在
            if len(seqs) > 1:
                # 混合批次，第一个是prefill，后续是decode
                # 为了不浪费算力，我们把那个正在做 Prefill 的“大胖子”放在 seqs[0]，然后把所有正在排队等着 Decode 的“瘦子”们塞在后面 seqs[1:].
                # prefill阶段不消耗token_ids
                prompt_tokens_left = seqs[0].num_prompt_tokens - seqs[0].prefilled_tokens - seqs[0].num_cached_tokens
                if prompt_tokens_left > 0:
                    new_chunk_size = min(self.chunk_size, prompt_tokens_left)
                    seqs[0].prefilled_tokens += new_chunk_size
                    # decode部分
                    for i, (seq, token_id) in enumerate(zip(seqs[1:], token_ids)):
                        self.block_manager.may_append(seq)
                        seq.append_token(token_id)
                        # 检查是否结束，结束则释放块并从运行队列移除
                        # 注意这里的结束条件包括遇到eos token或者达到最大生成长度
                        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                            seq.status = SequenceStatus.FINISHED
                            self.block_manager.deallocate(seq)
                            self.running.remove(seq)
            else:
                # 纯prefill，只更新prefilled_len
                prompt_tokens_left = seqs[0].num_prompt_tokens - seqs[0].prefilled_tokens - seqs[0].num_cached_tokens
                prefill_chunk_size = min(self.chunk_size, prompt_tokens_left)
                seqs[0].prefilled_tokens += prefill_chunk_size
        else:
            # 纯decode批次
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                # 检查是否结束，结束则释放块并从运行队列移除
                # 注意这里的结束条件包括遇到eos token或者达到最大生成长度
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

