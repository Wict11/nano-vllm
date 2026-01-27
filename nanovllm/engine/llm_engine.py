import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 从 kwargs 过滤出 Config 所需的参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 为多卡张量并行启动子进程设计的部分
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # 从1开始是因为这里起的是子进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
        # [ ] chunked prefill 大小参数
        self.chunked_prefill_size = config.chunked_prefill_size

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 如果是字符串则进行tokenization，把字符串转换为token id列表
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # 完整的prompt包括用户输入和采样参数，封装为Sequence对象
        seq = Sequence(prompt, sampling_params)
        # 把请求加入调度器
        self.scheduler.add(seq)

    def step(self):
        # [ ] 传入chunked_prefill_size参数
        # 调度器安排序列进行预填充或解码
        seqs, is_prefill, num_prefill_tokens, num_decode_tokens = self.scheduler.schedule()
        # 调用模型运行器执行计算
        token_ids = self.model_runner.call("run", seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
        # 调度器进行后处理
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 收集输出结果
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 计算本次步骤处理的 token 数量，用于吞吐量统计
        # 正数表示预填充阶段的 token 数量，负数表示解码阶段的 token 数量
        # num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        if is_prefill:
            num_tokens = 0
            for seq in seqs:
                prompt_tokens_left = len(seq) - seq.prefilled_tokens - seq.num_cached_tokens
                num_tokens += min(prompt_tokens_left, self.chunked_prefill_size)
            # BUG 这里重复计算了一次，应该是统计本次step实际处理的token数量
            # num_tokens = sum(seq.prefilled_tokens for seq in seqs)
        else:
            # Decode 阶段，每个序列只生成 1 个 token，用负数表示 Decode 吞吐
            num_tokens = -len(seqs)
            
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            # 进度条
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            # 每个prompt对应一个采样参数，如果只传入一个采样参数，但是有多份prompt，则复制多份采样参数
            # 广播采样参数
            sampling_params = [sampling_params] * len(prompts)

        # 把所有请求加入调度器
        for prompt, sp in zip(prompts, sampling_params):
            # 把任务放进了 Scheduler 的 waiting 队列里，并没有立即开始推理
            self.add_request(prompt, sp)
        
        # 初始化结果存储字典
        outputs = {}

        # 做吞吐量统计
        prefill_throughput = decode_throughput = 0.
        # 同步阻塞（Blocking） 的循环，直到所有序列都完成生成
        # is_finished 检查调度器的 waiting 队列和 running 队列是否都空了。
        while not self.is_finished():
            # 统计每次step的耗时，用于计算吞吐量
            t = perf_counter()
            # 步骤执行
            # step() 内部完成了：调度 -> 模型前向 -> 采样 -> 后处理
            output, num_tokens = self.step()

            # 添加计算TBT
            end_time = perf_counter()
            duration = end_time - t
            if num_tokens > 0:
                latency_stats["prefill_steps"].append(duration)
            else:
                latency_stats["decode_steps"].append(duration)

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 有一个请求彻底完成了，进度条才走一格
        # 重新排序输出结果，保证和输入顺序一致
        # 在并行推理中，短的 Prompt 或者生成短回复的任务会先结束。
        # 所以 output 吐出来的顺序和输入的 prompts 顺序是不一致的。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
