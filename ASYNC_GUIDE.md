# Nano-vLLM 异步流水线实现指南

## 目录
- [需求分析](#需求分析)
- [核心思路](#核心思路)
- [架构设计](#架构设计)
- [模块实现](#模块实现)
- [关键细节](#关键细节)
- [使用方式](#使用方式)
- [性能分析](#性能分析)

---

## 需求分析

### 背景问题

在原始的串行执行模式中，LLM 推理流程是同步阻塞的：

```
Step N:   [schedule N (0.5ms)] → [run N (50ms, GPU阻塞)] → [postprocess N (0.5ms)]
Step N+1: [schedule N+1 (0.5ms)] → [run N+1 (50ms, GPU阻塞)] → [postprocess N+1 (0.5ms)]
```

**存在的问题：**
1. **CPU 等待 GPU**：schedule 和 postprocess 必须等待 GPU 推理完成才能执行
2. **GPU 等待 CPU**：GPU 推理完成后，需要等待 CPU 完成 schedule 下一批次
3. **串行执行浪费**：CPU 调度和 GPU 推理无法并行，降低了整体吞吐量

### 核心需求

**目标：实现 vLLM v1 风格的异步调度流水线**

1. **CPU-GPU 并行**：CPU 调度和 GPU 推理应该流水线并行执行
2. **状态一致性**：在 chunked prefill 场景下，必须正确管理 pending 状态
3. **向后兼容**：保持与现有 API 的兼容性，用户可选择启用异步模式
4. **模块化设计**：异步逻辑独立封装，不影响串行模式

---

## 核心思路

### 理想的流水线执行

```
时间线对比：

串行模式（原始）：
CPU: [sched N] ----等待GPU---- [post N] [sched N+1] ----等待GPU---- [post N+1]
GPU:            [===== run N =====]                  [===== run N+1 =====]
时间: 51ms per step

异步模式（优化后）：
CPU: [post N-1][sched N] [post N][sched N+1] [post N+1][sched N+2]
GPU:            [==== run N ====][=== run N+1 ===][== run N+2 ==]
时间: 50ms per step（CPU 调度与 GPU 推理重叠）
```

### Chunked Prefill 的挑战

**状态依赖问题：**

```python
# schedule() 需要读取 prefilled_len 来决定调度哪个 chunk
prompt_remaining = seq.num_prompt_tokens - seq.prefilled_len

# postprocess() 更新 prefilled_len
seq.prefilled_len += chunk_size
```

**如果直接异步会出现什么问题？**

```
错误场景：
Step N:   schedule(N) 读取 prefilled_len=0 → 调度 chunk [0:512]
          run_async(N) 启动（GPU 异步执行中...）
          
Step N+1: schedule(N+1) 读取 prefilled_len=0  ← 还是 0！（postprocess 还没执行）
          → ❌ 重复调度 chunk [0:512]
```

### 解决方案：Pending 状态管理

**核心思想：**

维护两种状态：
- **Actual State（实际状态）**：已完成推理并处理的状态（`prefilled_len`）
- **Pending State（待定状态）**：已调度但推理未完成的状态（`pending_prefilled_len`）

```python
# 计算有效的 prefilled_len（包括 pending）
effective_prefilled = seq.prefilled_len + pending_prefilled_len
                      ↑                    ↑
                   已完成的              已调度但未完成的
```

**状态转换流程：**

```
Step N:
  schedule(N):
    effective = 0 + 0 = 0
    调度 chunk [0:512]
    记录到 pending: chunk_info[seq_id] = 512
  
  run_async(N): GPU 异步执行 chunk [0:512]...
  
Step N+1:
  schedule(N+1):
    effective = 0 + 512 = 512  ← 考虑 pending 状态
    调度 chunk [512:1024]
    记录到 pending: chunk_info[seq_id] = 512 (新的)
  
  postprocess(N):  ← N 的 GPU 推理完成了
    seq.prefilled_len = 512  ← 应用 pending → actual
    从 pending 移除已完成的批次
  
  run_async(N+1): GPU 异步执行 chunk [512:1024]...
```

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                         LLMEngine                            │
│  ┌────────────┐                                              │
│  │enable_async│ = False → Scheduler + ModelRunner (串行)     │
│  │ 参数选择    │ = True  → AsyncScheduler + AsyncModelRunner  │
│  └────────────┘                                              │
└─────────────────────────────────────────────────────────────┘

串行模式（原始）：
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Scheduler   │ -->  │ ModelRunner  │ -->  │ postprocess  │
│  (同步调度)   │      │  (阻塞执行)   │      │  (同步更新)   │
└──────────────┘      └──────────────┘      └──────────────┘

异步模式（新增）：
┌──────────────────┐      ┌────────────────────┐      ┌──────────────┐
│ AsyncScheduler   │ -->  │ AsyncModelRunner   │ -->  │ postprocess  │
│ (pending管理)     │      │ (CUDA Stream异步)  │      │ (pending→actual)│
│ effective_len    │      │ 非阻塞启动          │      │               │
└──────────────────┘      └────────────────────┘      └──────────────┘
```

### 模块职责

| 模块 | 职责 | 关键特性 |
|------|------|----------|
| **AsyncScheduler** | 调度管理 + Pending 状态追踪 | • 使用 `effective_prefilled_len` 调度<br>• 维护 `pending_batches` 队列<br>• postprocess 时应用 pending → actual |
| **AsyncModelRunner** | 异步推理执行 | • 使用 CUDA Stream 非阻塞启动<br>• 维护 `pending_results` 队列<br>• 提供 `wait_for_result()` 同步点 |
| **LLMEngine** | 流水线协调 | • 根据 `enable_async` 选择组件<br>• `_step_async()` 实现流水线逻辑<br>• 处理最后一批次的特殊逻辑 |
| **Sequence** | 序列状态 | • 添加 `aborted` 属性支持取消<br>• `prefilled_len` 为实际状态 |

---

## 模块实现

### 1. Sequence 类扩展

**文件：** `nanovllm/engine/sequence.py`

**修改内容：**

```python
class Sequence:
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # ... 原有代码 ...
        self.prefilled_len = 0  # 已完成的 prefill 长度
        self.aborted = False    # 标记序列是否被取消（异步调度使用）
```

**设计说明：**
- `aborted`：支持异步模式下的请求取消，postprocess 时过滤已取消的序列
- 保持向后兼容：串行模式不使用此字段

---

### 2. AsyncScheduler 实现

**文件：** `nanovllm/engine/async_scheduler.py`

**核心数据结构：**

```python
class AsyncScheduler:
    def __init__(self, config: Config):
        # 基础调度组件
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.block_manager = BlockManager(...)
        
        # 异步流水线支持
        self.pending_batches: deque[Dict] = deque()  # 追踪未完成的批次
        
        # 统计信息
        self.stats = {
            "total_scheduled": 0,
            "pending_batches": 0,
            "max_pending_batches": 0
        }
```

**关键方法 1：schedule() - 使用 effective_prefilled_len**

```python
def schedule(self) -> tuple[list[Sequence], bool, int, int]:
    """
    调度下一个批次
    
    关键区别：使用 effective_prefilled_len（包括 pending）
    """
    scheduled_seqs = []
    batch_chunk_info = {}  # seq_id -> chunk_size
    
    # Prefill 调度
    for seq in self.running:
        # ⭐ 关键：计算有效的 prefilled_len（包括 pending）
        effective_prefilled = self._get_effective_prefilled_len(seq)
        prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - effective_prefilled
        
        if prompt_remaining > 0:
            chunk_size = min(prompt_remaining, CHUNK_SIZE)
            scheduled_seqs.append(seq)
            batch_chunk_info[seq.seq_id] = chunk_size  # 记录 chunk 信息
            break
    
    # Decode 调度
    # ... (逻辑与串行类似，但使用 effective_prefilled_len)
    
    # ⭐ 记录 pending 批次信息
    self.pending_batches.append({
        'is_prefill': has_prefill,
        'chunk_info': batch_chunk_info,  # 用于 postprocess
        'seq_ids': [seq.seq_id for seq in scheduled_seqs]
    })
    
    return scheduled_seqs, has_prefill, num_prefill_tokens, num_decode_tokens
```

**关键方法 2：postprocess() - 应用 pending → actual**

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
    """
    处理推理结果
    
    关键：从 pending_batches 获取批次信息，应用 pending → actual 状态转换
    """
    # ⭐ 从 pending 队列获取批次信息（FIFO）
    if not self.pending_batches:
        raise RuntimeError("postprocess called but no pending batches")
    
    batch_info = self.pending_batches.popleft()
    chunk_info = batch_info['chunk_info']
    
    # 过滤已取消的序列
    active_seqs = [seq for seq in seqs if not seq.aborted]
    
    if is_prefill:
        if len(active_seqs) > 1:
            # 混合批次：第一个是 prefill，其余是 decode
            prefill_seq = active_seqs[0]
            if prefill_seq.seq_id in chunk_info:
                chunk_size = chunk_info[prefill_seq.seq_id]
                # ⭐ 应用 pending → actual
                prefill_seq.prefilled_len += chunk_size
            
            # 处理 decode 序列
            for seq, token_id in zip(active_seqs[1:], token_ids):
                self.block_manager.may_append(seq)
                seq.append_token(token_id)
                # ... 检查是否完成 ...
        else:
            # 纯 prefill
            prefill_seq = active_seqs[0]
            if prefill_seq.seq_id in chunk_info:
                chunk_size = chunk_info[prefill_seq.seq_id]
                prefill_seq.prefilled_len += chunk_size
    else:
        # 纯 decode
        for seq, token_id in zip(active_seqs, token_ids):
            seq.append_token(token_id)
            # ... 检查是否完成 ...
```

**关键方法 3：_get_effective_prefilled_len() - 计算有效长度**

```python
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
```

**设计亮点：**
1. **pending_batches 是 FIFO 队列**：保证 postprocess 按顺序应用状态
2. **chunk_info 记录精确信息**：支持混合批次和变长 chunk
3. **统计信息**：方便调试和性能分析

---

### 3. AsyncModelRunner 实现

**文件：** `nanovllm/engine/async_model_runner.py`

**核心设计：**

```python
class AsyncModelRunner:
    """
    异步模型执行器包装器
    
    包装标准 ModelRunner，添加异步执行支持
    """
    
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 初始化标准的 ModelRunner
        self.model_runner = ModelRunner(config, rank, event)
        
        # ⭐ 创建独立的推理 stream（只在主进程）
        if rank == 0:
            self.inference_stream = torch.cuda.Stream()
            self.pending_results = []  # [(result, event, args), ...]
            self.use_async = True
        else:
            self.use_async = False
        
        self.rank = rank
```

**关键方法 1：run_async() - 非阻塞启动**

```python
def run_async(self, seqs, is_prefill: bool, num_prefill_tokens: int, num_decode_tokens: int) -> None:
    """
    异步启动推理，立即返回（不等待完成）
    """
    if not self.use_async or self.rank != 0:
        # 非主进程直接同步执行
        result = self.model_runner.call("run", seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
        self.pending_results = [(result, None, None)]
        return
    
    # ⭐ 在独立 stream 中异步执行
    with torch.cuda.stream(self.inference_stream):
        # 执行推理
        result = self.model_runner.call("run", seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
        
        # ⭐ 创建同步事件
        event = torch.cuda.Event()
        event.record(self.inference_stream)
        
        # 记录 pending 结果
        self.pending_results.append((result, event, (seqs, is_prefill)))
    
    # ⭐ 立即返回，不等待 GPU 完成
```

**关键方法 2：wait_for_result() - 同步点**

```python
def wait_for_result(self) -> Optional[Any]:
    """
    等待最早的推理完成并返回结果
    
    Returns:
        推理结果（token_ids）
    """
    if not self.pending_results:
        return None
    
    result, event, args = self.pending_results.pop(0)
    
    # ⭐ 同步等待完成
    if event is not None:
        event.synchronize()  # 阻塞直到 GPU 推理完成
    
    return result
```

**设计亮点：**
1. **CUDA Stream 隔离**：使用独立 stream 避免阻塞默认 stream
2. **Event 同步机制**：精确控制 CPU-GPU 同步点
3. **包装器模式**：复用现有 ModelRunner，最小化代码修改
4. **多进程兼容**：非主进程回退到同步模式

---

### 4. LLMEngine 流水线协调

**文件：** `nanovllm/engine/llm_engine.py`

**初始化：根据 enable_async 选择组件**

```python
class LLMEngine:
    def __init__(self, model, enable_async: bool = False, **kwargs):
        """
        初始化 LLM 引擎
        
        Args:
            enable_async: 是否启用异步流水线模式（默认 False）
        """
        config = Config(model, **config_kwargs)
        
        # ⭐ 模式选择
        self.enable_async = enable_async
        
        # 启动子进程
        for i in range(1, config.tensor_parallel_size):
            runner_class = AsyncModelRunner if enable_async else ModelRunner
            process = ctx.Process(target=runner_class, args=(config, i, event))
            process.start()
            # ...
        
        # 主进程 ModelRunner
        if enable_async:
            self.model_runner = AsyncModelRunner(config, 0, self.events)
        else:
            self.model_runner = ModelRunner(config, 0, self.events)
        
        # 选择调度器
        if enable_async:
            self.scheduler = AsyncScheduler(config)
            self.pending_batch = None  # 追踪待处理的批次
            print("[LLMEngine] 异步流水线模式已启用")
        else:
            self.scheduler = Scheduler(config)
            print("[LLMEngine] 串行模式已启用")
```

**串行模式：_step_sync()**

```python
def _step_sync(self):
    """
    串行执行模式（原始逻辑）
    
    流程：schedule → run（阻塞） → postprocess
    """
    # 1. 调度
    seqs, is_prefill, num_prefill_tokens, num_decode_tokens = self.scheduler.schedule()
    
    # 2. 同步执行推理（阻塞等待）
    token_ids = self.model_runner.call("run", seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
    
    # 3. 处理结果
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    
    # 4. 收集输出
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    
    return outputs, num_tokens
```

**异步模式：_step_async() - 流水线核心**

```python
def _step_async(self):
    """
    异步流水线执行模式
    
    流程：
    1. 处理上一批次的结果（等待 GPU 完成）
    2. 调度下一批次（如果还有任务）
    3. 异步启动推理（立即返回，不等待）
    """
    outputs = []
    num_tokens = 0
    
    # ⭐ 步骤1: 处理上一批次的结果
    if self.pending_batch is not None:
        seqs, is_prefill = self.pending_batch
        
        # 等待推理完成并获取结果
        token_ids = self.model_runner.wait_for_result()
        
        if token_ids is not None:
            # 处理结果（应用 pending → actual）
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            
            # 收集输出
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            num_tokens = sum(seq.prefilled_len for seq in seqs) if is_prefill else -len(seqs)
        
        self.pending_batch = None
    
    # ⭐ 步骤2: 调度下一批次（检查是否还有任务）
    if self.scheduler.is_finished():
        return outputs, num_tokens
    
    seqs, is_prefill, num_prefill_tokens, num_decode_tokens = self.scheduler.schedule()
    
    # ⭐ 步骤3: 异步启动推理（立即返回）
    self.model_runner.run_async(seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
    
    # 记录待处理的批次
    self.pending_batch = (seqs, is_prefill)
    
    return outputs, num_tokens
```

**执行时间线对比：**

```
串行模式 step():
  [schedule] → [run (阻塞 50ms)] → [postprocess]
  总耗时: 51ms

异步模式 _step_async():
  Step N:   [wait_for_result (N-1)] [postprocess (N-1)] [schedule (N)] [run_async (N) 立即返回]
  Step N+1: [wait_for_result (N)]   [postprocess (N)]   [schedule (N+1)] [run_async (N+1)]
  
  CPU: [post N-1][sched N] [post N][sched N+1] (每次只需 1ms)
  GPU:            [== run N ==][= run N+1 =]   (重叠执行)
```

---

## 关键细节

### 细节 1: 最后一批次的处理

**问题：**

在 `generate()` 循环中，`is_finished()` 返回 `True` 后就退出循环，但异步模式下最后一个批次还在 `pending_batch` 中未处理。

**解决方案：**

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    # 添加请求
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    
    outputs = {}
    
    # 主循环
    while not self.is_finished():
        output, num_tokens = self.step()
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
            if use_tqdm:
                pbar.update(1)
    
    # ⭐ 异步模式下需要处理最后一个 pending 批次
    if self.enable_async and self.pending_batch is not None:
        seqs, is_prefill = self.pending_batch
        token_ids = self.model_runner.wait_for_result()
        if token_ids is not None:
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            for seq in seqs:
                if seq.is_finished and seq.seq_id not in outputs:
                    outputs[seq.seq_id] = seq.completion_token_ids
                    if use_tqdm:
                        pbar.update(1)
        self.pending_batch = None
    
    # 排序输出
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    return [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
```

**原因：**
- 异步模式的流水线设计中，`step()` 返回的是 **上一批次** 的输出
- 最后一次 `step()` 调度了最后一批但立即返回，还未等待完成
- 必须在循环外显式等待并处理最后一批

---

### 细节 2: 显存估算优化

**问题：**

在多次实例化 LLM 时（如示例中先运行串行再运行异步），第二次实例化会因为显存估算不准确而失败。

**原因：**

```python
# 旧的估算逻辑（不准确）
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
```

问题：`peak` 是峰值内存，可能包含已释放的内存，导致估算过小。

**解决方案：**

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    
    # ⭐ 清理缓存，避免上一个实例的缓存影响估算
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # 重新获取显存状态（清理后）
    free, total = torch.cuda.mem_get_info()
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    
    # ⭐ 使用更准确的公式
    # 可分配显存 = 总显存 × 利用率 - 当前已分配
    available_memory = int(total * config.gpu_memory_utilization) - current
    config.num_kvcache_blocks = available_memory // block_bytes
    
    # ⭐ 回退逻辑，避免断言失败
    if config.num_kvcache_blocks <= 0:
        config.num_kvcache_blocks = 1
        print("[WARN] 可用显存不足，已回退为1个KV块...")
```

**改进点：**
1. 清理前先 `synchronize()` 确保所有操作完成
2. 重置峰值统计，避免历史数据影响
3. 简化公式，不再使用混淆的 `used + peak - current`
4. 添加回退逻辑，避免直接失败

---

### 细节 3: 资源清理

**问题：**

示例脚本中先运行串行模式再运行异步模式，需要确保第一个实例的资源被完全释放。

**解决方案：**

```python
def example_sync():
    llm = LLM(model_path, max_model_len=2048, chunk_prefill_size=512)
    try:
        outputs = llm.generate(prompts, SamplingParams(max_tokens=20))
        print(f"\n输出: {outputs[0]['text']}")
    finally:
        # ⭐ 确保释放进程/显存资源，便于后续重新实例化
        llm.exit()

def example_async():
    llm = LLM(model_path, max_model_len=2048, chunk_prefill_size=512, enable_async=True)
    try:
        outputs = llm.generate(prompts, SamplingParams(max_tokens=20))
        print(f"\n输出: {outputs[0]['text']}")
    finally:
        # ⭐ 确保释放资源，防止与后续实例冲突
        llm.exit()
```

**LLMEngine.exit() 改进：**

```python
def exit(self):
    # ⭐ 防护：如果构造失败，model_runner 可能不存在
    if hasattr(self, "model_runner"):
        try:
            self.model_runner.call("exit")
        finally:
            del self.model_runner
    
    for p in self.ps:
        p.join()  # 等待子进程结束
```

---

### 细节 4: 空调度保护

**问题：**

异步模式下，postprocess 完成后可能所有序列都已完成，此时调度会返回空列表，导致 `assert scheduled_seqs` 失败。

**解决方案：**

```python
def _step_async(self):
    # ... postprocess 上一批次 ...
    
    # ⭐ 调度前检查是否还有任务
    if self.scheduler.is_finished():
        return outputs, num_tokens
    
    # 安全调度（此时必有任务）
    seqs, is_prefill, num_prefill_tokens, num_decode_tokens = self.scheduler.schedule()
    
    # ... 异步启动推理 ...
```

---

### 细节 5: Sequence.aborted 属性

**问题：**

AsyncScheduler 中使用了 `seq.aborted` 来过滤已取消的序列，但原始 Sequence 类没有这个属性。

**解决方案：**

```python
# nanovllm/engine/sequence.py
class Sequence:
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # ... 原有字段 ...
        self.prefilled_len = 0
        self.aborted = False  # ⭐ 添加取消标记
```

**使用场景：**

```python
# AsyncScheduler.postprocess()
active_seqs = [seq for seq in seqs if not seq.aborted]

# AsyncScheduler.abort_request()
for seq in list(self.running):
    if hasattr(seq, 'request_id') and seq.request_id == request_id:
        seq.aborted = True
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)
```

---

## 使用方式

### 基本用法

```python
from nanovllm import LLM, SamplingParams

# 串行模式（默认）
llm = LLM(
    model_path,
    max_model_len=2048,
    chunk_prefill_size=512
)

# 异步模式（启用流水线）
llm = LLM(
    model_path,
    max_model_len=2048,
    chunk_prefill_size=512,
    enable_async=True  # ← 唯一区别
)

# API 完全相同
outputs = llm.generate(prompts, SamplingParams(max_tokens=20))
```

### 完整示例

参见 `example_async_usage.py`：

```python
def example_sync():
    """串行模式示例（默认）"""
    llm = LLM("/path/to/model", max_model_len=2048, chunk_prefill_size=512)
    try:
        prompts = ["Hello, how are you?"]
        outputs = llm.generate(prompts, SamplingParams(max_tokens=20))
        print(f"\n输出: {outputs[0]['text']}")
    finally:
        llm.exit()

def example_async():
    """异步流水线模式示例"""
    llm = LLM("/path/to/model", max_model_len=2048, chunk_prefill_size=512, enable_async=True)
    try:
        prompts = ["Hello, how are you?"]
        outputs = llm.generate(prompts, SamplingParams(max_tokens=20))
        print(f"\n输出: {outputs[0]['text']}")
        # 查看调度统计
        if hasattr(llm.scheduler, 'get_stats'):
            print(f"\n调度统计: {llm.scheduler.get_stats()}")
    finally:
        llm.exit()

if __name__ == "__main__":
    example_sync()
    example_async()
```

### 关闭 Chunked Prefill

如果想关闭 chunked prefill（一次性 prefill 整个 prompt）：

```python
llm = LLM(
    model_path,
    max_model_len=2048,
    chunk_prefill_size=99999,  # 设置为很大的值
    enable_async=True
)
```

**注意：**
- 异步逻辑仍然正常工作（pending 状态只会记录一次完整 prefill）
- 但流水线收益会降低（prefill 只有一个批次）
- 确保 `max_num_batched_tokens` 足够容纳最长 prompt

---

## 性能分析

### 实际测试结果

```
============================================================
串行模式示例
============================================================
[LLMEngine] 串行模式已启用
Generating: 100%|██████| 1/1 [00:00<00:00, 1.13it/s, Prefill=7tok/s, Decode=274tok/s]

输出:  😊 Hello! I'm glad to hear that! 😊 How about you today...

============================================================
异步流水线模式示例
============================================================
[LLMEngine] 异步流水线模式已启用
Generating: 100%|██████| 1/1 [00:00<00:00, 8.41it/s, Prefill=1482tok/s, Decode=29020tok/s]

输出:  Yes, I am, as you and I. Also, I'd like to point out that the...

调度统计: {'total_scheduled': 21, 'pending_batches': 1, 'max_pending_batches': 1, 
          'waiting': 0, 'running': 0, 'current_pending': 0}
```

### 性能对比

| 指标 | 串行模式 | 异步模式 | 提升 |
|------|---------|---------|------|
| **Decode 吞吐** | 274 tok/s | 29020 tok/s | 105x（统计偏差） |
| **调度次数** | 21 | 21 | 相同 |
| **Max Pending** | N/A | 1 | 符合设计 |
| **API 兼容性** | 100% | 100% | 无改变 |

### 统计说明

**为什么 Decode 吞吐显示这么高？**

异步模式的吞吐统计有偏差：
```python
# 串行模式：计时包含完整推理
token_ids = self.model_runner.call("run", ...)  # 阻塞 50ms
decode_throughput = num_tokens / (perf_counter() - t)  # 真实时间

# 异步模式：计时只包含 CPU 操作
self.model_runner.run_async(...)  # 立即返回 < 1ms
decode_throughput = num_tokens / (perf_counter() - t)  # 极短时间 → 极高吞吐
```

**真实收益在哪里？**

1. **端到端延迟降低**：总耗时从 `51ms * 21 = 1071ms` 降低到约 `50ms * 21 + 调度时间`
2. **GPU 利用率提升**：GPU 无需等待 CPU 调度，连续推理
3. **复杂场景收益更明显**：
   - 多请求并发
   - 长 prompt + chunked prefill
   - 复杂调度策略

### 理论分析

**单请求场景（当前测试）：**

```
串行: 21 steps × 51ms = 1071ms
异步: 21 steps × 50ms = 1050ms
收益: ~2% （调度很快，重叠收益小）
```

**多请求场景（更复杂调度）：**

假设调度时间增加到 5ms：

```
串行: N steps × (5ms + 50ms + 5ms) = N × 60ms
异步: N steps × 50ms （调度与推理重叠）
收益: ~17%
```

**长 prompt + chunked prefill：**

假设 8192 token prompt，chunk_size=512，需要 16 个 prefill 步骤：

```
串行: 16 prefill × 51ms + M decode × 51ms
异步: 16 prefill × 50ms + M decode × 50ms（重叠执行）
收益: 每个 step 节省 1ms，总计节省 (16 + M) ms
```

---

## 调试和监控

### 调度统计

```python
# 获取调度统计信息
stats = llm.scheduler.get_stats()
print(stats)

# 输出：
# {
#   'total_scheduled': 21,        # 总共调度的批次数
#   'pending_batches': 1,         # 峰值 pending 批次数
#   'max_pending_batches': 1,     # 最大 pending 数
#   'waiting': 0,                 # 当前 waiting 队列长度
#   'running': 0,                 # 当前 running 队列长度
#   'current_pending': 0          # 当前 pending 批次数
# }
```

### 常见问题排查

**1. AssertionError: scheduled_seqs 为空**

原因：调度时没有任务可调度
解决：检查 `is_finished()` 调用，确保在有任务时才调度

**2. AttributeError: 'Sequence' object has no attribute 'aborted'**

原因：Sequence 类未添加 `aborted` 属性
解决：在 `Sequence.__init__()` 中添加 `self.aborted = False`

**3. 显存不足警告**

原因：显存估算不准确或实际剩余显存不足
解决：
- 降低 `gpu_memory_utilization`（如 0.8）
- 减小 `max_model_len` 或 `chunk_prefill_size`
- 确保上一个实例已调用 `exit()` 释放资源

**4. 输出不完整**

原因：异步模式下最后一批次未处理
解决：确保 `generate()` 循环后有处理 `pending_batch` 的逻辑

---

## 总结

### 实现要点

1. **Pending 状态管理**：通过 `effective_prefilled_len` 解决 chunked prefill 状态依赖
2. **CUDA Stream 异步**：使用独立 stream 实现非阻塞推理
3. **流水线协调**：LLMEngine 正确编排 postprocess → schedule → run_async
4. **模块化设计**：异步逻辑独立封装，不影响串行模式
5. **向后兼容**：通过 `enable_async` 参数选择，API 保持不变

### 适用场景

**推荐使用异步模式：**
- ✅ 多请求并发处理
- ✅ 长 prompt + chunked prefill
- ✅ 复杂调度策略（调度耗时较长）
- ✅ 对吞吐量要求高的场景

**可使用串行模式：**
- ✅ 简单单请求场景
- ✅ 短 prompt 无 chunked prefill
- ✅ 调度逻辑简单（< 1ms）
- ✅ 调试和开发阶段

### 未来优化方向

1. **多批次 Pending**：支持 pending 队列长度 > 1，进一步提升并发
2. **自适应调度**：根据 GPU 占用率动态调整 pending 深度
3. **更精细的统计**：区分 CPU 时间和 GPU 时间，准确测量收益
4. **Mixed-Precision Pipeline**：不同精度的 prefill 和 decode

---

## 参考资料

- [vLLM v1 Blog](https://blog.vllm.ai/2024/09/05/perf-update.html)
- [ASYNC_PIPELINE_DESIGN.md](./ASYNC_PIPELINE_DESIGN.md) - 详细设计方案
- [ASYNC_ARCHITECTURE.md](./ASYNC_ARCHITECTURE.md) - 架构说明
- PyTorch CUDA Stream 文档
