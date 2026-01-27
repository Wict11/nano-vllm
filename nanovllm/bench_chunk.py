"""
Token 生成时间线基准测试

记录每个请求每一步生成 token 的时间戳，并可视化：
- 横轴：时间 (ms)
- 纵轴：累积生成的 token 数量

这样可以直观看到：
1. 背景请求的 token 生成速度（斜率）
2. 长请求插入后，背景请求的生成是否变慢（斜率变缓）
"""

import os
import sys
import time
import random
import json
from typing import List, Dict, Any
import numpy as np

# 添加 nanovllm 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nano-vllm_code"))

from nanovllm import LLM, SamplingParams


def benchmark_single_config(
    llm,
    background_prompts: List[List[int]],
    long_prompts: List[List[int]],
    sampling_params: SamplingParams,
    config_name: str
) -> Dict[str, Any]:
    """
    记录每个请求每一步生成 token 的时间戳
    
    场景：
    1. 启动背景请求（短 prompt）
    2. 等待背景请求进入 decode 阶段
    3. 在第 10 步插入长请求
    4. 继续运行直到所有请求完成
    5. 记录每个请求的完整时间线和 TBT 数据
    """
    # 预热
    warmup_prompt = [[random.randint(0, 9999) for _ in range(50)]]
    llm.add_request(warmup_prompt[0], sampling_params)
    while not llm.is_finished():
        llm.step()
    
    # 记录数据结构
    request_timelines = {}  # seq_id -> {type, prompt_len, timeline: [(time_ms, tokens)]}
    tbt_data = {}          # seq_id -> [(token_idx, tbt_ms)]
    ttft_data = {}         # seq_id -> first token time (ms)
    ttft_data = {}         # seq_id -> first token time (ms)
    request_info = {}      # seq_id -> {type, injected_at_step, prompt_len}
    last_token_time = {}   # seq_id -> (time_sec, token_count)
    step_logs = []         # 每步的调度信息
    
    # 追踪请求
    bg_seq_ids = set()
    long_seq_ids = set()
    num_bg_started = 0
    num_long_started = 0
    
    def record_step_info(step_num, current_time):
        """记录当前步骤的调度信息"""
        step_info = {
            'step': step_num,
            'time': current_time,
            'running': [],
            'waiting': []
        }
        
        for seq in llm.scheduler.running:
            prompt_tokens = len(seq.prompt_token_ids)
            completion_tokens = len(seq.completion_token_ids)
            step_info['running'].append({
                'seq_id': seq.seq_id,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'is_prefill': completion_tokens == 0
            })
        
        for seq in llm.scheduler.waiting:
            prompt_tokens = len(seq.prompt_token_ids)
            completion_tokens = len(seq.completion_token_ids)
            step_info['waiting'].append({
                'seq_id': seq.seq_id,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'is_prefill': completion_tokens == 0
            })
        
        step_logs.append(step_info)
    
    # === 阶段 1: 启动背景请求并运行到 step 10 ===
    for i, prompt in enumerate(background_prompts):
        llm.add_request(prompt, sampling_params)
        num_bg_started += 1
    
    benchmark_start = time.perf_counter()
    step_count = 0
    INJECT_AT_STEP = 10
    
    def update_all_requests(current_time):
        """更新所有活跃请求的状态"""
        # 遍历 scheduler 中的所有活跃序列
        for seq in list(llm.scheduler.running) + list(llm.scheduler.waiting):
            seq_id = seq.seq_id
            
            # 首次见到的请求
            if seq_id not in request_timelines:
                # 根据当前步数判断类型
                if step_count <= INJECT_AT_STEP:
                    bg_seq_ids.add(seq_id)
                    req_type = "background"
                else:
                    long_seq_ids.add(seq_id)
                    req_type = "long"
                
                request_timelines[seq_id] = {
                    "prompt_len": len(seq.prompt_token_ids),
                    "type": req_type,
                    "timeline": []
                }
                request_info[seq_id] = {
                    'type': req_type,
                    'injected_at_step': 0 if req_type == "background" else step_count,
                    'prompt_len': len(seq.prompt_token_ids)
                }
                tbt_data[seq_id] = []
                last_token_time[seq_id] = None
            
            # 记录当前状态
            relative_time_ms = current_time * 1000
            token_count = len(seq.completion_token_ids)
            request_timelines[seq_id]["timeline"].append((relative_time_ms, token_count))
            if token_count > 0 and seq_id not in ttft_data:
                ttft_data[seq_id] = relative_time_ms
            
            # 计算 TBT
            if last_token_time[seq_id] is None:
                if token_count > 0:
                    last_token_time[seq_id] = (current_time, token_count)
            else:
                prev_time, prev_count = last_token_time[seq_id]
                if token_count > prev_count:
                    tbt_ms = (current_time - prev_time) * 1000
                    tbt_data[seq_id].append((token_count, tbt_ms))
                    last_token_time[seq_id] = (current_time, token_count)
    
    # 运行到 step 10
    while step_count < INJECT_AT_STEP:
        step_count += 1
        step_start = time.perf_counter()
        outputs, _ = llm.step()
        step_end = time.perf_counter()
        current_time = step_end - benchmark_start
        
        # 记录调度信息和所有请求状态
        record_step_info(step_count, current_time)
        update_all_requests(current_time)
    
    # === 阶段 2: 插入长请求 ===
    for i, prompt in enumerate(long_prompts):
        llm.add_request(prompt, sampling_params)
        num_long_started += 1
    
    # === 阶段 3: 继续运行直到所有请求完成 ===
    
    while not llm.is_finished():
        step_count += 1
        step_start = time.perf_counter()
        outputs, _ = llm.step()
        step_end = time.perf_counter()
        current_time = step_end - benchmark_start
        
        # 记录调度信息和所有请求状态
        record_step_info(step_count, current_time)
        update_all_requests(current_time)
    
    total_time = time.perf_counter() - benchmark_start
    
    # 打印请求信息摘要
    print(f"\n{'='*80}")
    print("Request Summary:")
    print(f"{'='*80}")
    print(f"  Total requests tracked: {len(request_timelines)}")
    print(f"  Background: {len(bg_seq_ids)}, Long: {len(long_seq_ids)}")
    print()
    print(f"{'Seq ID':<8} {'Type':<12} {'Prompt':<8} {'Generated':<10} {'TTFT(ms)':<12} {'Avg TPOT':<12} {'Max TPOT':<12}")
    print("-" * 90)
    
    for seq_id in sorted(request_timelines.keys()):
        data = request_timelines[seq_id]
        info = request_info.get(seq_id, {})
        tbts = tbt_data.get(seq_id, [])
        
        if data["timeline"]:
            total_tokens = data["timeline"][-1][1]
            ttft_ms = ttft_data.get(seq_id)
            avg_tpot = np.mean([t[1] for t in tbts]) if tbts else None
            max_tpot = max([t[1] for t in tbts]) if tbts else None
            ttft_str = f"{ttft_ms:.2f}" if ttft_ms is not None else "N/A"
            avg_tpot_str = f"{avg_tpot:.2f}" if avg_tpot is not None else "N/A"
            max_tpot_str = f"{max_tpot:.2f}" if max_tpot is not None else "N/A"
            print(f"{seq_id:<8} {data['type']:<12} {info.get('prompt_len', 0):<8} "
                  f"{total_tokens:<10} {ttft_str:<12} {avg_tpot_str:<12} {max_tpot_str:<12}")
    
    # 打印关键步骤的调度信息
    print(f"\n{'='*80}")
    print(f"Scheduling Info (Steps {max(1, INJECT_AT_STEP-2)} to {min(len(step_logs), INJECT_AT_STEP+10)}):")
    print(f"{'='*80}")
    
    for log in step_logs[max(0, INJECT_AT_STEP-3):min(len(step_logs), INJECT_AT_STEP+10)]:
        print(f"\nStep {log['step']} @ {log['time']:.3f}s:")
        if log['running']:
            print(f"  Running ({len(log['running'])}):")
            for seq_info in log['running']:
                status = "PREFILL" if seq_info['is_prefill'] else "DECODE"
                print(f"    - Seq {seq_info['seq_id']}: {status:7s} | "
                      f"Prompt: {seq_info['prompt_tokens']:4d}, Generated: {seq_info['completion_tokens']:3d}")
        if log['waiting']:
            print(f"  Waiting ({len(log['waiting'])}):")
            for seq_info in log['waiting']:
                status = "PREFILL" if seq_info['is_prefill'] else "DECODE"
                print(f"    - Seq {seq_info['seq_id']}: {status:7s} | "
                      f"Prompt: {seq_info['prompt_tokens']:4d}, Generated: {seq_info['completion_tokens']:3d}")
    
    return {
        "config_name": config_name,
        "total_time_ms": total_time * 1000,
        "num_background": len(background_prompts),
        "num_long": len(long_prompts),
        "request_timelines": request_timelines,
        "tbt_data": tbt_data,
        "ttft_data": ttft_data,
        "request_info": request_info,
        "step_logs": step_logs,
        "background_prompts": [len(p) for p in background_prompts],
        "long_prompts": [len(p) for p in long_prompts],
    }


def plot_timeline(result: Dict[str, Any], save_path: str = None):
    """绘制包含两子图的总图：时间线 + TPOT"""
    import matplotlib.pyplot as plt
    
    request_timelines = result["request_timelines"]
    tbt_data = result.get("tbt_data", {})
    
    bg_color = "#3498db"
    long_color = "#e74c3c"
    
    fig, (ax_timeline, ax_tpot) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1：时间线
    for seq_id in sorted(request_timelines.keys()):
        data = request_timelines[seq_id]
        timeline = data["timeline"]
        if not timeline:
            continue
        times = [t for t, _ in timeline]
        tokens = [n for _, n in timeline]
        color = bg_color if data["type"] == "background" else long_color
        marker = 's' if data["type"] == "background" else 'o'
        ax_timeline.plot(times, tokens, marker=marker, markersize=3, linewidth=1.8,
                         label=f"{data['type'].title()} seq_{seq_id}", color=color, alpha=0.85)
    ax_timeline.set_xlabel("Time (ms)", fontsize=12, fontweight='bold')
    ax_timeline.set_ylabel("Cumulative Tokens", fontsize=12, fontweight='bold')
    ax_timeline.set_title(f"Timeline - {result['config_name']}", fontsize=14, fontweight='bold')
    ax_timeline.legend(fontsize=9, loc='upper left')
    ax_timeline.grid(True, alpha=0.3, linestyle='--')
    
    # 子图2：TPOT 曲线
    for seq_id in sorted(request_timelines.keys()):
        data = request_timelines[seq_id]
        tbts = tbt_data.get(seq_id, [])
        if not tbts:
            continue
        token_ids = [t for t, _ in tbts]
        tpot_ms = [v for _, v in tbts]
        color = bg_color if data["type"] == "background" else long_color
        marker = 's' if data["type"] == "background" else 'o'
        ax_tpot.plot(token_ids, tpot_ms, marker=marker, markersize=3, linewidth=1.8,
                     label=f"{data['type'].title()} seq_{seq_id}", color=color, alpha=0.85)
    ax_tpot.set_xlabel("Token Index", fontsize=12, fontweight='bold')
    ax_tpot.set_ylabel("TPOT (ms)", fontsize=12, fontweight='bold')
    ax_tpot.set_title(f"Per-Token Latency - {result['config_name']}", fontsize=14, fontweight='bold')
    ax_tpot.legend(fontsize=9, loc='upper right')
    ax_tpot.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main(model_path: str, chunk_size: int, save_results: str = None, save_plot: str = None):
    """主函数"""
    
    # 配置名称
    if chunk_size >= 999999:
        config_name = f"Chunked Prefill DISABLED (chunk_size={chunk_size})"
    else:
        config_name = f"Chunked Prefill ENABLED (chunk_size={chunk_size})"
    
    # 生成测试 prompts
    vocab_size = 10000
    
    # 2 个背景请求：短 prompt (~20 tokens)
    background_prompts = [
        [random.randint(0, vocab_size - 1) for _ in range(random.randint(15, 25))]
        for _ in range(5)
    ]
    
    # 1 个长请求：长 prompt (~1000 tokens)
    long_prompts = [
        [random.randint(0, vocab_size - 1) for _ in range(random.randint(980, 1020))]
        for _ in range(5)
    ]
    
    llm = LLM(model_path, max_model_len=8192, chunk_prefill_size=chunk_size)
    
    # 采样参数
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.0,  # greedy
    )
    
    # 运行 benchmark
    result = benchmark_single_config(
        llm, background_prompts, long_prompts, 
        sampling_params, config_name
    )
    
    # 保存结果
    if save_results:
        with open(save_results, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to {save_results}")
    
    # 绘制图表
    if save_plot:
        plot_timeline(result, save_plot)
    elif not save_results:
        # 如果没指定保存路径，直接显示
        plot_timeline(result)
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Token generation timeline benchmark")
    parser.add_argument("--model", type=str, 
                       default="/mnt/workspace/nano_vllm/nano-vllm/Qwen/Qwen3-0.6B",
                       help="Path to model")
    parser.add_argument("--chunk-size", type=int, default=999999, 
                       help="Chunk size for prefill (999999 to disable)")
    parser.add_argument("--save-results", type=str, default=None, 
                       help="Save results to JSON file")
    parser.add_argument("--save-plot", type=str, default="timeline_disable.png", 
                       help="Save plot to image file")
    args = parser.parse_args()
    
    main(args.model, args.chunk_size, args.save_results, args.save_plot)