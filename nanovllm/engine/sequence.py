from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


# 定义序列的三种状态，用于引擎调度
class SequenceStatus(Enum):
    WAITING = auto() # 序列等待被调度执行
    RUNNING = auto() # 序列正在被模型执行
    FINISHED = auto() # 序列已经完成生成


class Sequence:
    # 用于内存分页管理。每个块包含 256 个 tokens，便于 GPU 内存分配和 KV 缓存优化。
    block_size = 256
    # 用于生成唯一的序列 ID
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter) # 分配唯一的序列 ID
        self.status = SequenceStatus.WAITING # 初始状态为等待
        self.token_ids = copy(token_ids) # 深拷贝，避免外部修改
        self.last_token = token_ids[-1] 
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0 # 已缓存的 token 数量
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        
        # [ ] 已经prefill的token数量
        self.prefilled_tokens = 0 # 已经prefill的token数量

    def __len__(self):
        return self.num_tokens # 返回序列的总 token 数量

    def __getitem__(self, key):
        return self.token_ids[key] # 支持通过索引访问 token ID
    
    @property # 属性装饰器，将方法作为属性访问
    def is_finished(self): # 判断序列是否已完成生成
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self): # 计算生成的 token 数量
        return self.num_tokens - self.num_prompt_tokens

    @property # token切片
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self): # 计算序列包含的块数量
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # [ ] 获取下一个chunk的 token 切片
    def get_next_prefill_chunk(self):
        """
            获取下一个预填充的 token 切片
        """
        start = self.prefilled_tokens
        end = min(self.prefilled_tokens + self.chunked_prefill_size, self.num_prompt_tokens)
        chunk = self.token_ids[start:end]
        return chunk, end-start

    def block(self, i):
        """
            返回第 i 个块的 token 切片
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # 添加新生成的 token 到序列末尾
    def append_token(self, token_id: int): 
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 返回序列的状态元组，包括 token 数、缓存信息和 token_ids（如果没有补全）
    # 或最后一个 token（如果有补全）。
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
