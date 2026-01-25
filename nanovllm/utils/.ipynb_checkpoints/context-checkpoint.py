from dataclasses import dataclass

import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0


_CONTEXT = Context()


def get_context():
    return _CONTEXT


# def set_context(
#     is_prefill,
#     cu_seqlens_q=None,
#     cu_seqlens_k=None,
#     max_seqlen_q=0,
#     max_seqlen_k=0,
#     slot_mapping=None,
#     context_lens=None,
#     block_tables=None,
# ):
#     global _CONTEXT
#     _CONTEXT = Context(
#         is_prefill,
#         cu_seqlens_q,
#         cu_seqlens_k,
#         max_seqlen_q,
#         max_seqlen_k,
#         slot_mapping,
#         context_lens,
#         block_tables,
#     )
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, num_prefill_tokens=0, num_decode_tokens=0):
    global _CONTEXT
    # _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, num_prefill_tokens, num_decode_tokens)
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
