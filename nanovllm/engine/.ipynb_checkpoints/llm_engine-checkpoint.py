import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.async_scheduler import AsyncScheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.async_model_runner import AsyncModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

latency_stats = {
    "prefill_steps": [], # 记录每次 prefill step 的耗时
    "decode_steps": []   # 记录每次 decode step 的耗时
}

class LLMEngine:

    def __init__(self, model, enable_async: bool = False, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        self.enable_async = enable_async  # 添加异步
        
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            runner_class = AsyncModelRunner if enable_async else ModelRunner # 选择modelRunner类型
            process = ctx.Process(target=runner_class, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 这个是主进程的ModelRunner，负责tensor并行的第0个分片
        
        if enable_async:
            self.model_runner = AsyncModelRunner(config, 0, self.events)
        else:
            self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        # 根据模式选择调度器
        if enable_async:
            self.scheduler = AsyncScheduler(config)
            self.pending_batch = None  # 追踪待处理的批次（异步模式）
            print("[LLMEngine] 异步流水线模式已启用")
        else:
            self.scheduler = Scheduler(config)
            print("[LLMEngine] 串行模式已启用")
            
        atexit.register(self.exit)
        self.chunk_size = config.chunk_prefill_size
    
    def exit(self):
        if hasattr(self, "model_runner"):
            try:
                self.model_runner.call("exit")
            finally:
                del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        
    def step(self):
        '''
        执行一步推理
        
        根据 enable_async 参数选择执行模式：
        - False（串行模式）：schedule → run（阻塞） → postprocess
        - True（异步模式）：postprocess(N-1) → schedule(N) → run_async(N)（非阻塞）
        
        Returns:
            (outputs, num_tokens): 已完成序列的输出和token数量
        '''
        if self.enable_async:
            return self._step_async()
        else:
            return self._step_sync()
    
    def _step_sync(self):
        '''
        串行执行模式（原始逻辑）
        
        流程：schedule → run（阻塞） → postprocess
        '''
        # -------------------- chunked prefill逻辑 --------------------
        # 1. 获取混合批次及元数据
        t0 = perf_counter()
        seqs, is_prefill, num_prefill_tokens, num_decode_tokens = self.scheduler.schedule()
        # 2. 同步执行推理（阻塞等待）
        t1 = perf_counter()
        print(f"Scheduling time: {(t1 - t0) * 1000:.2f}ms, is_prefill={is_prefill}, num_prefill_tokens={num_prefill_tokens}, num_decode_tokens={num_decode_tokens}, num_seqs={len(seqs)}")
        token_ids = self.model_runner.call("run", seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
        # 3. 处理结果
        t2 = perf_counter()
        print(f"Model run time: {(t2 - t1) * 1000:.2f}ms")
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        t3 = perf_counter()
        print(f"Postprocess time: {(t3 - t2) * 1000:.2f}ms")
        
        # 4. 收集输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 5. 计算 token 数量
        if is_prefill:
            CHUNK_SIZE = self.chunk_size
            num_tokens = sum(seq.prefilled_len for seq in seqs)
        else:
            num_tokens = -len(seqs)
        
        return outputs, num_tokens
    
    def _step_async(self):
        '''
        异步流水线执行模式
        
        流程：
        1. 处理上一批次的结果（等待 GPU 完成）
        2. 调度下一批次（要判断是否还有任务）
        3. 异步启动推理（立即返回，不等待）
        '''
        outputs = []
        num_tokens = 0
        
        # -------------------- 步骤1: 处理上一批次 --------------------
        if self.pending_batch is not None:
            seqs, is_prefill = self.pending_batch
            
            # 等待推理完成并获取结果
            token_ids = self.model_runner.wait_for_result()
            
            if token_ids is not None:
                # 处理结果
                self.scheduler.postprocess(seqs, token_ids, is_prefill)
                
                # 收集输出
                outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
                
                # 计算 token 数量
                if is_prefill:
                    num_tokens = sum(seq.prefilled_len for seq in seqs)
                else:
                    num_tokens = -len(seqs)
            
            self.pending_batch = None
        
        # -------------------- 步骤2: 调度下一批次 --------------------
        # 检查是否还有任务需要调度
        if self.scheduler.is_finished():
            return outputs, num_tokens
        seqs, is_prefill, num_prefill_tokens, num_decode_tokens = self.scheduler.schedule()
        
        # -------------------- 步骤3: 异步启动推理 --------------------
        self.model_runner.run_async(seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
        
        # 记录待处理的批次
        self.pending_batch = (seqs, is_prefill)
        
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
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # ------------------添加计算TBT-------------------
            end_time = perf_counter()
            duration = end_time - t
            if num_tokens > 0:
                # prefill阶段，记录每次prefill step的耗时
                latency_stats["prefill_steps"].append(duration)
            else:
                # decode阶段，记录每次decode step的耗时
                latency_stats["decode_steps"].append(duration)
            # ------------------添加计算TBT-------------------
            if use_tqdm:
                # 异步模式下，step() 立即返回不等待 GPU，时间测量不准确
                # 因此不显示即时吞吐量，避免误导（只在最后显示准确的总体吞吐）
                if not self.enable_async:
                    if num_tokens > 0: # prefill阶段，计算吞吐
                        prefill_throughput = num_tokens / (perf_counter() - t)
                    else:              # decode阶段算吞吐
                        decode_throughput = -num_tokens / (perf_counter() - t)
                    pbar.set_postfix({ 
                        # 更新进度条的后缀信息，显示prefill和decode的吞吐
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    })
                else:
                    # 异步模式：只显示模式标识，不显示即时吞吐量
                    pbar.set_postfix({"Mode": "Async Pipeline"})
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # 异步模式下需要处理最后一个 pending 批次
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

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
