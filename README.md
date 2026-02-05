# Nano-vLLM

基于nana-vLLM，实现一些功能。
* **支持Qwen3、Qwen-MoE、LLama2的模型**
* **添加chunked prefill功能**
* **添加异步调度功能**
* **其他功能实现中。。**

## Chunked Prefill 设计逻辑

* 🚀 **Scheduler layer** - 为长序列切分chunk并添加至调度队列，优先级依次是：running队列中的prefill阶段序列、waiting队列中的序列、running队列中decode阶段的序列
* 📖 **LLM engine layer** - 额外传入num_prefill_tokens和num_decode_tokens数据，区分混合prefill和decode的批次
* 💡 **Attention layer** - 针对混合批次，分别调用flash attn的函数接口来处理，最后合并数据并返回
* 💡 **Post progress** - 只有decode阶段序列要计算logits和更新产生的token

## 异步调度设计逻辑
**详情请看ASYNC_GUIDE.md文件**

## Installation

```bash
pip install git+https://github.com/Wict11/nano-vllm.git
```

## Manual Download

If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

**0.Test Configuration:**
- Hardware: A10 (24GB)
- Model: Qwen3-0.6B
- Total Requests: 3 sequences（for test）
- short background flows(~20tokens): 5
- long incast flows(~1000tokens): 5

**1.chunked prefill性能结果：**
* **流量情况：**
五个背景短流（20tokens）
五个incast长流（1000tokens）
* **Disabled Chunked Prefill:**
<img width="1198" height="437" alt="image" src="https://github.com/user-attachments/assets/953d9f9f-c954-4bd3-8d6e-602a14f8e981" />

<img width="679" height="226" alt="image" src="https://github.com/user-attachments/assets/3d7a35a9-7d87-4cbc-a9aa-d0b250618f9e" />

* **Chunk_size = 512:**
  <img width="1188" height="438" alt="image" src="https://github.com/user-attachments/assets/93109a4d-580f-4f01-991c-36cf22909430" />

  <img width="736" height="201" alt="image" src="https://github.com/user-attachments/assets/4c8d25f2-900a-4152-a163-916c192f0281" />
* **结果分析：**
  对混合批次的decode序列性能影响：开chunked prefill后，最大TPOT减少～4.8x，
  对混合批次的prefill序列性能影响：开chunked prefill后，TTFT至多增加～1.5x
  
* **待做测试：最大可支持长度**
  不开chunked prefill最大能支持的序列长度为：xxx
  开了chunked prefill最大能支持的序列长度为：xxx，提升xxx倍
  
**2.异步调度性能结果：**
。。。



