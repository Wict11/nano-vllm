from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False # 当前是否处于预填充阶段
    cu_seqlens_q: torch.Tensor | None = None # 查询（query）的累积序列长度
    cu_seqlens_k: torch.Tensor | None = None # 键（key）的累积序列长度
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 槽映射（slot mapping），
    # 用于分页注意力中的内存槽分配，帮助将逻辑序列映射到物理内存块。
    slot_mapping: torch.Tensor | None = None
    # 上下文长度张量，表示每个序列的上下文（提示）长度，用于区分提示和生成部分
    context_lens: torch.Tensor | None = None
    # 块表（block tables），用于分页注意力，记录每个序列的块分配情况，支持 KV 缓存的复用和内存分页。
    block_tables: torch.Tensor | None = None

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, num_prefill_tokens=0, num_decode_tokens=0):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill, 
        cu_seqlens_q, 
        cu_seqlens_k, 
        max_seqlen_q, 
        max_seqlen_k, 
        slot_mapping, 
        context_lens, 
        block_tables, 
        num_prefill_tokens = num_prefill_tokens, 
        num_decode_tokens = num_decode_tokens)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
