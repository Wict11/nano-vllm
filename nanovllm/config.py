import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass # 自动生成构造函数和默认值。
class Config:
    model: str # 模型路径
    max_num_batched_tokens: int = 16384 # 单次批量推理允许的最大 token 数量
    max_num_seqs: int = 512 # 单次批量推理允许的最大序列数量
    max_model_len: int = 4096 # 模型的最大上下文长度
    gpu_memory_utilization: float = 0.9 # GPU 显存使用率上限，用于动态内存分配
    tensor_parallel_size: int = 1 # 张量并行度大小 1 表示单卡，>1 表示分布式切分参数
    enforce_eager: bool = False # 是否强制使用 eager 模式（非编译模式）
    hf_config: AutoConfig | None = None # HuggingFace AutoConfig 对象，初始化后会自动加载
    eos: int = -1 # 模型的 end-of-sequence token ID，默认 -1 可稍后修改
    kvcache_block_size: int = 256 # KV cache 块大小，必须是 256 的倍数
    num_kvcache_blocks: int = -1 # KV cache 块数量，-1 表示动态计算
    
    # [ ] chunked prefill 相关参数
    chunked_prefill_size: int = 1024  # chunked prefill 的大小，单位为 token 数量

    # 做 参数验证和初始化衍生值
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
