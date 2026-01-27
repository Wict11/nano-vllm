import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 把x按照最后一个维度（列）切分成两半，因为一半是gate计算结果，一半是up计算结果
        x, y = x.chunk(2, -1)
        # 左边做silu激活，右边直接返回，然后对应元素相乘
        return F.silu(x) * y
