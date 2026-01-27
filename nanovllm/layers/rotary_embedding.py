from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 可以看Rope论文公式来理解这部分
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int, # 128
        rotary_dim: int, # 128
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 每一对 (x_{2i}, x_{2i+1}) 对应一个频率
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)) # 一共64个值
        t = torch.arange(max_position_embeddings, dtype=torch.float) # 40960
        # 外积，每个位置 × 每个频率
        freqs = torch.einsum("i,j -> ij", t, inv_freq) # (40960, 64)
        cos = freqs.cos()
        sin = freqs.sin()
        # unsqueeze：在指定维度插入一个大小为 1 的新维度，为了 后面和 Q/K 对齐时能 broadcast
        # rope_cache[positions] → (B*S, 1, head_dim)，可以自动 broadcast
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1) # (40960, 1, 128)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        # 分开成 cos 和 sin
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
