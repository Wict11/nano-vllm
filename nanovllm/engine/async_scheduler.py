"""
异步调度器 - 支持 CPU-GPU 流水线并行执行

核心特性：
1. Pending 状态管理 - 解决 chunked prefill 状态依赖
2. 与串行调度器接口一致
3. 支持流水线重叠执行
"""

from collections import deque
from typing import Optional, Tuple, List, Dict

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class AsyncScheduler:
    """
    异步调度器 - 支持流水线并行
    
    与 Scheduler 的区别：
    1. 维护 pending 状态（已调度但未完成的批次）
    2. 使用 effective_prefilled_len 调度（包括 pending）
    3. postprocess 时应用 pending → actual 状态转换
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.chunk_size = config.chunk_prefill_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 调度队列
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        
        # 异步流水线支持
        self.pending_batches: deque[Dict] = deque()  # 追踪未完成的批次信息
        
        # 统计信息
        self.stats = {
            "total_scheduled": 0,
            "pending_batches": 0,
            "max_pending_batches": 0
        }

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, int, int]:
        """
        调度下一个批次
        
        关键区别：使用 effective_prefilled_len（包括 pending）
        
        Returns:
            (seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
        """
        CHUNK_SIZE = self.chunk_size
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        has_prefill = False
        
        # 记录本批次的 chunk 信息（用于 postprocess）
        batch_chunk_info = {}  # seq_id -> chunk_size
        
        # =============== Prefill 调度 ===============
        # 检查 running 队列中未完成 prefill 的序列
        prefill_seq_found = None
        for seq in list(self.running):
            if num_seqs >= self.max_num_seqs:
                break
            
            # 关键：使用 effective_prefilled_len（包括 pending）
            effective_prefilled = self._get_effective_prefilled_len(seq)
            prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - effective_prefilled
            
            if prompt_remaining <= 0:
                continue  # 已经 prefill 完成
            
            this_chunk_size = min(prompt_remaining, CHUNK_SIZE)
            
            if num_batched_tokens + this_chunk_size > self.max_num_batched_tokens:
                break
            
            num_seqs += 1
            num_batched_tokens += this_chunk_size
            scheduled_seqs.append(seq)
            has_prefill = True
            prefill_seq_found = seq
            
            # 记录 chunk 信息
            batch_chunk_info[seq.seq_id] = this_chunk_size
            
            break  # 每次只处理一个 prefill 序列的一个 chunk
        
        # 从 running 队列中临时移除已调度的 prefill 序列
        if prefill_seq_found is not None:
            self.running.remove(prefill_seq_found)
        
        # 从 waiting 队列添加新的 prefill 请求
        if not has_prefill and self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # 使用 effective_prefilled_len
            effective_prefilled = self._get_effective_prefilled_len(seq)
            prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - effective_prefilled
            this_chunk_size = min(prompt_remaining, CHUNK_SIZE)
            
            if this_chunk_size > 0 and \
               num_batched_tokens + this_chunk_size <= self.max_num_batched_tokens and \
               self.block_manager.can_allocate(seq):
                
                num_seqs += 1
                self.block_manager.allocate(seq)
                
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += this_chunk_size
                has_prefill = True
                
                # 记录 chunk 信息
                batch_chunk_info[seq.seq_id] = this_chunk_size

        # =============== Decode 调度 ===============
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # 检查是否可以 append
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # 跳过已调度为 prefill 的序列
                if seq in scheduled_seqs:
                    continue
                
                # 检查是否完成 prefill
                effective_prefilled = self._get_effective_prefilled_len(seq)
                prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - effective_prefilled
                
                if prompt_remaining > 0:
                    self.running.appendleft(seq)
                    continue
                
                # Decode 阶段
                if self.block_manager.can_append(seq) and \
                   num_batched_tokens + 1 <= self.max_num_batched_tokens:
                    
                    # 纯 decode 批次才提前 append
                    if not has_prefill:
                        self.block_manager.may_append(seq)
                    
                    num_batched_tokens += 1
                    num_seqs += 1
                    scheduled_seqs.append(seq)
                    
                    # Decode 不需要记录 chunk_size
                else:
                    self.running.appendleft(seq)
                    break

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        
        # 记录 pending 批次信息
        self.pending_batches.append({
            'is_prefill': has_prefill,
            'chunk_info': batch_chunk_info,
            'seq_ids': [seq.seq_id for seq in scheduled_seqs]
        })
        
        # 更新统计
        self.stats["total_scheduled"] += 1
        self.stats["pending_batches"] = len(self.pending_batches)
        self.stats["max_pending_batches"] = max(
            self.stats["max_pending_batches"],
            len(self.pending_batches)
        )
        
        # 计算 token 数量
        num_prefill_tokens = 0
        num_decode_tokens = 0
        for seq in scheduled_seqs:
            if seq.seq_id in batch_chunk_info:
                num_prefill_tokens += batch_chunk_info[seq.seq_id]
            else:
                num_decode_tokens += 1
        
        return scheduled_seqs, has_prefill, num_prefill_tokens, num_decode_tokens

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool = False):
        """
        处理推理结果
        
        关键区别：从 pending_batches 获取批次信息，应用 pending → actual 状态转换
        """
        # 从 pending 队列获取批次信息
        if not self.pending_batches:
            raise RuntimeError("postprocess called but no pending batches")
        
        batch_info = self.pending_batches.popleft()
        chunk_info = batch_info['chunk_info']
        
        # 过滤已取消的序列
        active_seqs = [seq for seq in seqs if not getattr(seq, 'aborted', False)]
        
        if not active_seqs:
            return
        
        CHUNK_SIZE = self.chunk_size
        
        if is_prefill:
            if len(active_seqs) > 1:
                # 混合批次
                prefill_seq = active_seqs[0]
                
                # 应用 pending → actual
                if prefill_seq.seq_id in chunk_info:
                    chunk_size = chunk_info[prefill_seq.seq_id]
                    prefill_seq.prefilled_len += chunk_size
                
                # 处理 decode 序列
                for seq, token_id in zip(active_seqs[1:], token_ids):
                    if seq.aborted:
                        continue
                    
                    self.block_manager.may_append(seq)
                    seq.append_token(token_id)
                    
                    if (not seq.ignore_eos and token_id == self.eos) or \
                       seq.num_completion_tokens == seq.max_tokens:
                        seq.status = SequenceStatus.FINISHED
                        self.block_manager.deallocate(seq)
                        self.running.remove(seq)
            else:
                # 纯 prefill
                prefill_seq = active_seqs[0]
                
                # 应用 pending → actual
                if prefill_seq.seq_id in chunk_info:
                    chunk_size = chunk_info[prefill_seq.seq_id]
                    prefill_seq.prefilled_len += chunk_size
        else:
            # 纯 decode
            for seq, token_id in zip(active_seqs, token_ids):
                if seq.aborted:
                    continue
                
                seq.append_token(token_id)
                
                if (not seq.ignore_eos and token_id == self.eos) or \
                   seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

    def preempt(self, seq: Sequence):
        """抢占序列"""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def abort_request(self, request_id: str):
        """取消请求"""
        # 从 waiting 队列移除
        self.waiting = deque([
            seq for seq in self.waiting 
            if not (hasattr(seq, 'request_id') and seq.request_id == request_id)
        ])
        
        # 从 running 队列移除并释放资源
        for seq in list(self.running):
            if hasattr(seq, 'request_id') and seq.request_id == request_id:
                seq.aborted = True
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                break

    def _get_effective_prefilled_len(self, seq: Sequence) -> int:
        """
        计算有效的 prefilled 长度（包括 pending 状态）
        
        这是异步调度的关键：考虑已调度但未完成的 chunk
        """
        actual_prefilled = seq.prefilled_len
        
        # 累计所有 pending 批次中该序列的 chunk_size
        pending_prefilled = 0
        for batch_info in self.pending_batches:
            chunk_info = batch_info.get('chunk_info', {})
            if seq.seq_id in chunk_info:
                pending_prefilled += chunk_info[seq.seq_id]
        
        return actual_prefilled + pending_prefilled

    def get_stats(self) -> dict:
        """获取调度器统计信息"""
        return {
            **self.stats,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "current_pending": len(self.pending_batches)
        }
