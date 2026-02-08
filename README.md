# Nano-vLLM

基于nana-vLLM，实现一些功能。
* **支持Qwen3、Qwen-MoE、LLama2的模型**
* **添加chunked prefill功能**
* **添加异步调度功能**
* **其他功能实现中。。**

## Chunked Prefill 设计逻辑

* 🚀 **Scheduler layer** - 为长序列切分chunk并添加至调度队列，优先级依次是：running队列中的prefill阶段序列、waiting队列中的序列、running队列中decode阶段的序列
* 📖 **LLM engine layer** - 额外传入num_prefill_tokens和num_decode_tokens数据，区分混合prefill和decode的批次
* 🧠 **Attention layer** - 针对混合批次，分别调用flash attn的函数接口来处理，最后合并数据并返回
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

**🌟1.chunked prefill性能结果：**

* **脚本：base_chunk_v4.py**

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
  
**🌟2.异步调度性能结果：**

* **脚本：bench_async.py**

* **多小流并发场景：**

* ***🎬场景一:128条800token流量并发***

  *  **串行结果：**

    <img width="786" height="46" alt="image" src="https://github.com/user-attachments/assets/5d96c136-0809-4fd6-af3e-936315f8ac2e" />

  * **并行结果：**

    <img width="778" height="56" alt="image" src="https://github.com/user-attachments/assets/1ce72271-c234-43b5-97e5-249ada23b6e9" />

  * **结果分析：**
      从13288tok/s优化至14161tok/s，**6.5%左右提升**

* ***🎬场景二:128条800token流量并发 + chunked prefill***
  *  **chunk size: 512tok**  

  *  **串行结果：**

    <img width="776" height="33" alt="image" src="https://github.com/user-attachments/assets/6645fc44-8371-406e-a96f-ac79499151af" />

  * **并行结果：**

    <img width="768" height="60" alt="image" src="https://github.com/user-attachments/assets/d17b478b-e538-4c5d-b756-b8534e049508" />

  * **结果分析：**
      从8411tok/s优化至8752tok/s，**4.0%左右提升**

      开了chunked prefill吞吐会下降是因为目前只加一个chunk进入混合批次，没跑满最大单批次可添加的token数，后续再修改为根据总的token预算来
    





