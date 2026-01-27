import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
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
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

# 高效并行化
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # N个线程，每个线程处理一个token的key/value存储
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

def store_kvcache_simplified(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    flat_key = key.view(N, -1) # (N, num_heads * head_dim)
    flat_value = value.view(N, -1) # (N, num_heads * head_dim)

    # 根据slot_mapping把key/value存到对应位置
    for i in range(N):
        slot = slot_mapping[i].item()
        k_cache[slot] = flat_key[i]
        v_cache[slot] = flat_value[i]

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
        # 初始化空的k_cache和v_cache
        self.k_cache = self.v_cache = torch.tensor([])

#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
#         context = get_context()
#         k_cache, v_cache = self.k_cache, self.v_cache

#         if k_cache.numel() and v_cache.numel():
#             store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

#         if context.is_prefill:
#             is_mixed = (
#                 hasattr(context, 'num_prefill_tokens') and 
#                 hasattr(context, 'num_decode_tokens') and
#                 context.num_prefill_tokens is not None and 
#                 context.num_decode_tokens is not None and
#                 context.num_prefill_tokens > 0 and 
#                 context.num_decode_tokens > 0
#             )

#             if is_mixed:
#                 # 混合批次：prefill + decode
#                 num_prefill = context.num_prefill_tokens
#                 num_decode = context.num_decode_tokens

#                 q_prefill = q[:num_prefill]
#                 q_decode = q[num_prefill:]
#                 k_prefill = k[:num_prefill]
#                 v_prefill = v[:num_prefill]

#                 output_parts = []

#                 # ===== Part A: Prefill =====
#                 # 使用 varlen 处理 prefill 部分（因为需要处理历史 + 当前 chunk）
#                 # 构建 cu_seqlens 用于 varlen attention
#                 cu_seqlens_prefill = torch.tensor([0, num_prefill], device=q.device, dtype=torch.int32)

#                 o_prefill = flash_attn_varlen_func(
#                     q_prefill, k_prefill, v_prefill,
#                     cu_seqlens_q=cu_seqlens_prefill,
#                     cu_seqlens_k=cu_seqlens_prefill,
#                     max_seqlen_q=num_prefill,
#                     max_seqlen_k=num_prefill,
#                     softmax_scale=self.scale,
#                     causal=True
#                 )
#                 output_parts.append(o_prefill)

#                 # ===== Part B: Decode =====
#                 if q_decode.numel() > 0 and k_cache.numel() and v_cache.numel():
#                     # decode 部分使用 with_kvcache（因为只有 1 个新 token，需要读历史 cache）
#                     o_decode = flash_attn_with_kvcache(
#                         q_decode.unsqueeze(1),  # [num_decode, 1, heads, dim]
#                         k_cache, v_cache,
#                         block_table=context.block_tables[1:] if context.block_tables is not None else None,
#                         cache_seqlens=context.context_lens[1:] if context.context_lens is not None else None,
#                         softmax_scale=self.scale,
#                         causal=True
#                     ).squeeze(1)  # 回到 [num_decode, heads, dim]
#                     output_parts.append(o_decode)

#                 # 合并结果回到 [total_tokens, heads, dim]
#                 o = torch.cat(output_parts, dim=0)

#             else:
#                 # 纯 Prefill
#                 if k_cache.numel() and v_cache.numel() and context.block_tables is not None:
#                     # 有历史 KV 缓存（后续 chunks 的情况）
#                     o = flash_attn_with_kvcache(
#                         q.unsqueeze(0),  # [1, seq_len, heads, dim]
#                         k_cache, v_cache,
#                         block_table=context.block_tables,
#                         cache_seqlens=context.context_lens,
#                         softmax_scale=self.scale,
#                         causal=True
#                     ).squeeze(0)  # 回到 [seq_len, heads, dim]
#                 else:
#                     # 无 KV 缓存（第一个 chunk / warmup）
#                     o = flash_attn_varlen_func(
#                         q, k, v,
#                         cu_seqlens_q=context.cu_seqlens_q,
#                         cu_seqlens_k=context.cu_seqlens_k,
#                         max_seqlen_q=context.max_seqlen_q,
#                         max_seqlen_k=context.max_seqlen_k,
#                         softmax_scale=self.scale,
#                         causal=True
#                     )

#         else:
#             # === Pure Decode ===
#             o = flash_attn_with_kvcache(
#                 q.unsqueeze(1),
#                 k_cache, v_cache,
#                 cache_seqlens=context.context_lens,
#                 block_table=context.block_tables,
#                 softmax_scale=self.scale,
#                 causal=True
#             ).squeeze(1)

#         o = o.view(-1, self.num_heads * self.head_dim)
#         return o
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            is_mixed = (
                hasattr(context, 'num_prefill_tokens') and 
                hasattr(context, 'num_decode_tokens') and
                context.num_prefill_tokens is not None and 
                context.num_decode_tokens is not None and
                context.num_prefill_tokens > 0 and 
                context.num_decode_tokens > 0
            )

            if is_mixed:
                # 混合批次：prefill + decode
                num_prefill = context.num_prefill_tokens
                num_decode = context.num_decode_tokens

                q_prefill = q[:num_prefill]
                q_decode = q[num_prefill:]

                output_parts = []

                # ===== Part A: Prefill =====
                # 关键修改：也用 with_kvcache 来读历史 cache！
                if k_cache.numel() and v_cache.numel() and context.block_tables is not None:
                    # 有历史 KV 缓存，使用 with_kvcache 读取
                    o_prefill = flash_attn_with_kvcache(
                        q_prefill.unsqueeze(0),  # [1, num_prefill, heads, dim]
                        k_cache, v_cache,
                        block_table=context.block_tables[0:1],  # 取第 0 个序列的 block table
                        cache_seqlens=context.context_lens[0:1],  # 取第 0 个序列的总长度
                        softmax_scale=self.scale,
                        causal=True
                    ).squeeze(0)  # 回到 [num_prefill, heads, dim]
                else:
                    # 无历史缓存（不应该出现在混合批次中）
                    # 但保险起见，用 varlen 处理
                    cu_seqlens_prefill = torch.tensor([0, num_prefill], device=q.device, dtype=torch.int32)
                    o_prefill = flash_attn_varlen_func(
                        q_prefill, k[:num_prefill], v[:num_prefill],
                        cu_seqlens_q=cu_seqlens_prefill,
                        cu_seqlens_k=cu_seqlens_prefill,
                        max_seqlen_q=num_prefill,
                        max_seqlen_k=num_prefill,
                        softmax_scale=self.scale,
                        causal=True
                    )

                output_parts.append(o_prefill)

                # ===== Part B: Decode =====
                if q_decode.numel() > 0 and k_cache.numel() and v_cache.numel():
                    # decode 部分使用 with_kvcache
                    o_decode = flash_attn_with_kvcache(
                        q_decode.unsqueeze(1),  # [num_decode, 1, heads, dim]
                        k_cache, v_cache,
                        block_table=context.block_tables[1:] if context.block_tables is not None else None,
                        cache_seqlens=context.context_lens[1:] if context.context_lens is not None else None,
                        softmax_scale=self.scale,
                        causal=True
                    ).squeeze(1)  # 回到 [num_decode, heads, dim]
                    output_parts.append(o_decode)

                # 合并结果
                o = torch.cat(output_parts, dim=0)

            else:
                # 纯 Prefill
                if k_cache.numel() and v_cache.numel() and context.block_tables is not None:
                    # 有历史 KV 缓存（chunked prefill 的后续 chunks）
                    o = flash_attn_with_kvcache(
                        q.unsqueeze(0),  # [1, seq_len, heads, dim]
                        k_cache, v_cache,
                        block_table=context.block_tables,
                        cache_seqlens=context.context_lens,
                        softmax_scale=self.scale,
                        causal=True
                    ).squeeze(0)  # 回到 [seq_len, heads, dim]
                else:
                    # 无 KV 缓存（第一个 chunk / warmup）
                    o = flash_attn_varlen_func(
                        q, k, v,
                        cu_seqlens_q=context.cu_seqlens_q,
                        cu_seqlens_k=context.cu_seqlens_k,
                        max_seqlen_q=context.max_seqlen_q,
                        max_seqlen_k=context.max_seqlen_k,
                        softmax_scale=self.scale,
                        causal=True
                    )

        else:
            # === Pure Decode ===
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            ).squeeze(1)

        o = o.view(-1, self.num_heads * self.head_dim)
        return o