import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 调整 logits 的温度
        # logits大小是batch_size * 词库大小，temperature大小是 batch_size * 1，所以需要对第二个维度做广播复制
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) 
        # 调节温度之后做 softmax 和采样
        probs = torch.softmax(logits, dim=-1)
        # 做随机采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
