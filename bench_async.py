"""
异步流水线模式使用示例
"""

import os
import sys
import random
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nano-vllm_code"))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


def make_random_prompt(length: int = 64, vocab_size: int = 10000) -> list[int]:
    return [random.randint(1, vocab_size - 1) for _ in range(length)]


# 可调节的测试参数
PROMPT_LEN = 800       # 每条请求的输入长度（token 数）
NUM_PROMPTS = 128       # 请求数量
MAX_TOKENS = 100       # 每条请求生成的最大 token 数


def run_warmup(llm):
    """执行一次轻量 warmup，触发 CUDA 上下文/内核缓存"""
    print("进行一次 warmup...")
    llm.generate([make_random_prompt(8)], SamplingParams(max_tokens=1), use_tqdm=False)


def run_bandwidth(llm, prompts, sampling_params):
    """循环直到所有序列完成，仅统计吞吐，不打印生成结果"""
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)

    seqs = []
    for prompt, sp in zip(prompts, sampling_params):
        seq = Sequence(prompt, sp)
        seqs.append(seq)
        llm.scheduler.add(seq)

    t0 = perf_counter()

    while not llm.is_finished():
        llm.step()

    # 异步模式需要处理最后一个 pending 批次
    if getattr(llm, "enable_async", False) and getattr(llm, "pending_batch", None) is not None:
        seqs_pending, is_prefill = llm.pending_batch
        token_ids = llm.model_runner.wait_for_result()
        if token_ids is not None:
            llm.scheduler.postprocess(seqs_pending, token_ids, is_prefill)
        llm.pending_batch = None

    elapsed = perf_counter() - t0
    prompt_tokens = sum(seq.num_prompt_tokens for seq in seqs)
    generated_tokens = sum(seq.num_completion_tokens for seq in seqs)
    total_tokens = prompt_tokens + generated_tokens
    print(
        f"完成: prompt={prompt_tokens} tok, generated={generated_tokens} tok, "
        f"total={total_tokens} tok, 用时={elapsed:.3f}s, 吞吐={total_tokens/elapsed:.1f} tok/s"
    )


def example_sync():
    """串行模式示例（默认）"""
    print("=" * 60)
    print("串行模式示例")
    print("=" * 60)
    
    # 初始化（默认 enable_async=False）
    llm = LLM(
        "/mnt/workspace/nano_vllm/nano-vllm/Qwen/Qwen3-0.6B",
        max_model_len=2048,
        chunk_prefill_size=512
    )
    try:
        # 先执行一次轻量 warmup，避免首轮冷启动抖动
        run_warmup(llm)
        # 生成
        prompts = [make_random_prompt(PROMPT_LEN) for _ in range(NUM_PROMPTS)]
        run_bandwidth(llm, prompts, SamplingParams(max_tokens=MAX_TOKENS))
    finally:
        # 确保释放进程/显存资源，便于后续重新实例化
        llm.exit()


def example_async():
    """异步流水线模式示例"""
    print("\n" + "=" * 60)
    print("异步流水线模式示例")
    print("=" * 60)
    
    # 初始化（显式启用异步）
    llm = LLM(
        "/mnt/workspace/nano_vllm/nano-vllm/Qwen/Qwen3-0.6B",
        max_model_len=2048,
        chunk_prefill_size=512,
        enable_async=True  # ← 启用异步流水线
    )
    try:
        # 先执行一次轻量 warmup，避免首轮冷启动抖动
        run_warmup(llm)
        # 生成（API完全相同）
        prompts = [make_random_prompt(PROMPT_LEN) for _ in range(NUM_PROMPTS)]
        run_bandwidth(llm, prompts, SamplingParams(max_tokens=MAX_TOKENS))
    finally:
        # 确保释放资源，防止与后续实例冲突
        llm.exit()


if __name__ == "__main__":
    # 串行模式
    # example_sync()
    
    # 异步模式
    example_async()
