import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
        #     if context.block_tables is not None:  # prefix cache
        #         k, v = k_cache, v_cache
        #     o = flash_attn_varlen_func(
        #         q,
        #         k,
        #         v,
        #         max_seqlen_q=context.max_seqlen_q,
        #         cu_seqlens_q=context.cu_seqlens_q,
        #         max_seqlen_k=context.max_seqlen_k,
        #         cu_seqlens_k=context.cu_seqlens_k,
        #         softmax_scale=self.scale,
        #         causal=True,
        #         block_table=context.block_tables,
        #     )
        # 检查是否是混合批次（prefill + decode）
            # is_mixed = context.num_prefill_tokens > 0 and context.num_decode_tokens > 0
            # 需要确保字段存在且有效
            is_mixed = (
                hasattr(context, 'num_prefill_tokens') and 
                hasattr(context, 'num_decode_tokens') and
                context.num_prefill_tokens is not None and 
                context.num_decode_tokens is not None and
                context.num_prefill_tokens > 0 and 
                context.num_decode_tokens > 0
            )
                        # DEBUG
            # print(f"[DEBUG attention] is_mixed={is_mixed}, num_prefill={getattr(context, 'num_prefill_tokens', 'N/A')}, num_decode={getattr(context, 'num_decode_tokens', 'N/A')}, q.shape={q.shape}")
            
            
            if is_mixed:
                # 混合批次：分别处理 prefill 和 decode，然后合并
                num_prefill = context.num_prefill_tokens
                num_decode = context.num_decode_tokens
                
                # 分离 prefill 和 decode 部分
                q_prefill = q[:num_prefill]
                q_decode = q[num_prefill:]
                k_prefill = k[:num_prefill]
                v_prefill = v[:num_prefill]
                
                # 处理 prefill 部分 - 使用 varlen
                # 构建 prefill 的 cu_seqlens
                # 处理 prefill 部分 - 使用 varlen
                # 构建 prefill 的 cu_seqlens，基于实际的 prefill token 数量
                cu_seqlens_prefill = torch.tensor([0, num_prefill], device=q.device, dtype=torch.int32)
                o_prefill = flash_attn_varlen_func(
                    q_prefill, k_prefill, v_prefill,
                    max_seqlen_q=num_prefill,
                    cu_seqlens_q=cu_seqlens_prefill,
                    max_seqlen_k=num_prefill,
                    cu_seqlens_k=cu_seqlens_prefill,
                    softmax_scale=self.scale,
                    causal=True
                )
                
                # 处理 decode 部分 - 使用 kvcache
                # decode 序列从 cache 读取历史 KV
                o_decode = flash_attn_with_kvcache(
                    q_decode.unsqueeze(1),  # (num_decode, 1, num_heads, head_dim)
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens[1:] if context.context_lens is not None else None,
                    block_table=context.block_tables[1:] if context.block_tables is not None else None,
                    softmax_scale=self.scale,
                    causal=True
                ).squeeze(1)
                
                # 合并结果
                o = torch.cat([o_prefill, o_decode], dim=0)
            else:
                # 纯 prefill 批次
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                # o = flash_attn_varlen_func(q, k, v,
                #                            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                #                            max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                #                            softmax_scale=self.scale, causal=True, block_table=context.block_tables)
                # 注意：varlen 不接受 block_table 参数
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale, causal=True
                )
        else:  # decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
