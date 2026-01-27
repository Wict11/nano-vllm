import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 计算“我的”显卡序号
        self.tp_rank = dist.get_rank()
        # 显卡总数
        self.tp_size = dist.get_world_size()
        # 确保词表总数能被显卡数量整除，否则不好平均分
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        # 计算每个显卡负责的词数量
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 计算“我的”管辖范围是从第几个词开始，到第几个词结束。
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 只申请了 `num_embeddings_per_partition` (1/4 大小) 的显存
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    # 对词表做权重切分
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 判断输入的 token 是否在“我的”管辖范围内
            # 这里mask是一个布尔张量或者0/1 整数张量，形状和 x 一样，
            # 如果 x 是 [Batch_Size, Seq_Len]，那么 mask 也是 [Batch_Size, Seq_Len]
            # x 通常是一个二维张量：[Batch_Size, Seq_Len]（比如 [2, 1024]，表示 2 句话，每句 1024 个词）。
            # 或者是压扁的一维张量：[Total_Tokens]。
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 偏移索引
            # 假设Rank 0 负责词 ID 0~49。输入 x：[10, 80] （10 归我管，80 不归我管）
            # 把那些“不归我管”的越界索引，强制修改成一个合法的索引（通常是 0），防止查表时报错。
            # x = [1, 0] \times [10, 80] = [10, 0]
            x = mask * (x - self.vocab_start_idx)
        # 查表，把原来的每个整数 ID 替换成了一个长长的向量
        # weight 的形状是 [num_embeddings_per_partition, embedding_dim]即[Vocab_Part, Hidden]
        # x：[Batch, Seq] ->  y: [Batch, Seq, Hidden] 
        y = F.embedding(x, self.weight)
        # 多卡情况才需要执行“切分-掩码-同步”流程
        if self.tp_size > 1:
            # 局部结果归零
            # 需要把 mask 扩展到和 y 一样有 3 个维度，并且让它覆盖 Hidden 那个维度
            # mask 需要 unsqueeze 变成 [batch, 1] 才能广播乘 [batch, hidden]
            # 相乘 (Broadcasting)： mask 的 [2, 1] 会自动沿着 Hidden 维度复制，变成 [2, Hidden]
            y = mask.unsqueeze(1) * y
            # 把所有显卡上的 y 矩阵按元素相加 (Sum)，然后把结果同步给所有人
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        # 如果是多卡的话，需要把各个卡上的 logits 拼接起来
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
