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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 在qwen3中的attention部分，qkv已经变成了 ：
        # q = q.view(-1, self.num_heads, self.head_dim) 
        # k = k.view(-1, self.num_kv_heads, self.head_dim) 
        # v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 这样的多头形状
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            # 1. 存KV Cache
            # 把decode产生的token的新的 k, v 存到 k_cache, v_cache 里
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # 2. 准备参数
        # flash_attn_with_kvcache 需要 q 是 [Batch, SeqLen, Heads, Dim]
        # 但输入的 q 是被 flatten 过的 [Total_Tokens, Heads, Dim]
        # 我们需要把它还原回去，或者利用 unsqueeze 伪装成 Batch 维度
        
        # 这里的关键是：对于 Chunked Prefill，q 是一段长序列；对于 Decode，q 是长度为 1 的序列。
        # 只要我们正确设置 cache_seqlens，算子就能自动找到对应的历史。

        if context.is_prefill:
            # 检查是否是混合批次（prefill + decode）
            # 需要确保字段存在且有效
            is_mixed = (
                hasattr(context, 'num_prefill_tokens') and 
                hasattr(context, 'num_decode_tokens') and
                context.num_prefill_tokens is not None and 
                context.num_decode_tokens is not None and
                context.num_prefill_tokens > 0 and 
                context.num_decode_tokens > 0
            )
            if is_mixed:
                # 混合批次：分别处理 prefill 和 decode，然后合并
                num_prefill = context.num_prefill_tokens
                num_decode = context.num_decode_tokens
                
                # 分离 prefill 和 decode 部分
                q_prefill = q[:num_prefill]
                q_decode = q[num_prefill:]

                # k_prefill = k[:num_prefill]
                # v_prefill = v[:num_prefill]

                output_parts = []

                # Part A: Prefill (关键修改!)
                # 我们把它当成 batch=1, seqlen=num_prefill 来算
                # 这样它就能通过 block_table[0] 读到完整的历史

                # 代码默认纯 Prefill 阶段通常只有一个序列（Batch Size = 1）
                # 大部分简易推理引擎在 Prefill 阶段一次只允许处理一个 Prompt。
                o_prefill = flash_attn_with_kvcache(
                    q_prefill.unsqueeze(0), # [1, Chunk_Len, Heads, Dim] ---> [Batch_Size, Seq_Len, Num_Heads, Head_Dim]
                    k_cache, v_cache,
                    # 只取第0个序列的 block table，和q_prefill对应
                    block_table=context.block_tables[0:1], 
                    # 取第0个序列的总长度 (历史+当前)
                    cache_seqlens=context.context_lens[0:1],
                    softmax_scale=self.scale,
                    causal=True
                ).squeeze(0)
                output_parts.append(o_prefill)
                
                # # 处理 prefill 部分 - 使用 varlen
                # # 构建 prefill 的 cu_seqlens
                # # 处理 prefill 部分 - 使用 varlen
                # # 构建 prefill 的 cu_seqlens，基于实际的 prefill token 数量
                # cu_seqlens_prefill = torch.tensor([0, num_prefill], device=q.device, dtype=torch.int32)
                # o_prefill = flash_attn_varlen_func(
                #     q_prefill, k_prefill, v_prefill,
                #     max_seqlen_q=num_prefill,
                #     cu_seqlens_q=cu_seqlens_prefill,
                #     max_seqlen_k=num_prefill,
                #     cu_seqlens_k=cu_seqlens_prefill,
                #     softmax_scale=self.scale,
                #     causal=True,
                #     # [ ] prefix cache 支持
                #     block_table=context.block_tables if context.block_tables is not None else None
                # )

                # Part B: Decode (如果有)
                if q_decode.numel() > 0:
                    o_decode = flash_attn_with_kvcache(
                        q_decode.unsqueeze(1), # [Batch_Decode, 1, Heads, Dim]
                        k_cache, v_cache,
                        block_table=context.block_tables[1:], # 跳过第0个序列 
                        cache_seqlens=context.context_lens[1:],
                        softmax_scale=self.scale,
                        causal=True
                    ).squeeze(1)
                    output_parts.append(o_decode)
                
                o = torch.cat(output_parts, dim=0)

                # # 处理 decode 部分 - 使用 kvcache
                # # decode 序列从 cache 读取历史 KV
                # o_decode = flash_attn_with_kvcache(
                #     q_decode.unsqueeze(1),  # (num_decode, 1, num_heads, head_dim)
                #     k_cache,
                #     v_cache,
                #     cache_seqlens=context.context_lens[1:] if context.context_lens is not None else None,
                #     block_table=context.block_tables[1:] if context.block_tables is not None else None,
                #     softmax_scale=self.scale,
                #     causal=True
                # ).squeeze(1)
                
                # # 合并结果
                # o = torch.cat([o_prefill, o_decode], dim=0)
            else:
                # 纯 Prefill，但为了连上 Cache 里的历史，依然用 with_kvcache
                # 这里假设 Batch=1
                # 如果你在 Scheduler 里允许同时塞进去两个 Prompt 做 Prefill（比如 input_ids 是两个 Prompt 拼起来的），这里就会有问题
                o = flash_attn_with_kvcache(
                    q.unsqueeze(0), # [1, Seq_Len, Dim]
                    k_cache, v_cache,
                    block_table=context.block_tables, # [1, Max_Blocks] # 本来就只有1行，直接传
                    cache_seqlens=context.context_lens,
                    softmax_scale=self.scale,
                    causal=True
                ).squeeze(0)
                # # 纯 prefill 批次
                # if context.block_tables is not None:    # prefix cache
                #     k, v = k_cache, v_cache
                # # o = flash_attn_varlen_func(q, k, v,
                # #                            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                # #                            max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                # #                            softmax_scale=self.scale, causal=True, block_table=context.block_tables)
                # # 注意：varlen 不接受 block_table 参数
                # o = flash_attn_varlen_func(
                #     q, k, v,
                #     max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                #     max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                #     softmax_scale=self.scale, causal=True
                # )
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