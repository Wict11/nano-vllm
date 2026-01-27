import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        # num_heads ≠ num_kv_heads → 这是 GQA / MQA（Qwen3 默认是 GQA）
        num_heads: int,
        num_kv_heads: int,
        # RoPE 支持的最大上下文长度
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        # Q / K / V 线性层是否带 bias（Qwen 系列通常不开）
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        # tp并行数量
        tp_size = dist.get_world_size()

        # 每张 GPU 只负责一部分 Query heads
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        # Q heads 多，KV heads 少 → KV cache 更小
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        # 计算单个头的维度
        self.head_dim = head_dim or hidden_size // self.total_num_heads

        # 每个 GPU 负责的 Q / K / V 大小
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Attention 的缩放因子 sqrt(d_k)
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # QKV 投影层
        # 一次性并行生成 Q / K / V（支持 KV head 少于 Q head）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # Attention 结果 → hidden_size 的输出投影（Output Projection）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # RoPE 位置编码模块，在 Q / K 上旋转注入位置信息
        # RoPE（Rotary Positional Embedding）模块
        self.rotary_emb = get_rope(
            self.head_dim, # 每个 head 的维度
            rotary_dim=self.head_dim, # 整个 head 都用 RoPE
            max_position=max_position, # 支持最大序列长度
            base=rope_theta, # 频率基数
            rope_scaling=rope_scaling, # 长上下文扩展（Qwen3 必须）
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor, # 每个 token 的 绝对位置索引
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 投影到 q, k, v
        qkv = self.qkv_proj(hidden_states)
        # 拆分 Q / K / V
        # q: (B*S, num_heads * head_dim)
        # k: (B*S, num_kv_heads * head_dim)
        # v: (B*S, num_kv_heads * head_dim)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # reshape 成多头结构
        # 变成 (batch, num_heads, head_dim)的形状，用来算rmsnorm
        # 因为后面要：对 每个 head：做 RMSNorm、做 RoPE、做 Attention
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 只给qk做，v不做，为了稳定 QK 点积的尺度，减少 attention logit 漂移
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 注入位置编码，形状不变
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        # 拼接多头输出：(B*S, num_heads, head_dim) → (B*S, num_heads * head_dim)
        output = self.o_proj(o.flatten(1, -1))
        return output

# 门控，多层感知机（前馈网络）
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # 一般把gate和up合并成一个大矩阵一起乘，按列切分，混合起来计算
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2, # gate 和 up 都是 intermediate_size，两个矩阵合并在一起就是2倍
            bias=False,
        )
        # down投影矩阵，用于降维，按行切分计算
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        # 对单gpu上gate-up合并块 做 切块和计算
        self.act_fn = SiluAndMul()

    def forward(self, x): 
        # 门控、升维计算
        gate_up = self.gate_up_proj(x)
        # 激活
        x = self.act_fn(gate_up)
        # 降维
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 如果没有残差则表示是第一个子层
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        # 否则表示有残差连接
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        # 继续后续子层
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# (Base Model)，只包含：Embedding + N 层 Transformer Block + Norm。
class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 堆叠多个解码器层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 把输入的 token ids 转换为嵌入向量
        hidden_states = self.embed_tokens(input_ids)
        # 第一层的残差是 None
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后做一次归一化，只要一个输出，不需要残差
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# (Task Model)，包含 Base Model + LM Head。
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        # 基础模型
        self.model = Qwen3Model(config)
        # 语言模型头
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # lm_head（模型的出口，将词映射为概率） 和 embed_tokens（模型的入口，将 token 映射为嵌入向量） 共享权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor, # token ids
        positions: torch.Tensor, # 位置编码
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    # 计算概率分布
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
