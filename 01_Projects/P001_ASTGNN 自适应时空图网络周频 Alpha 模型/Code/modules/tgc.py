import torch
from torch import nn
from torch.nn import functional as F


# 从股票节点嵌入学习邻接矩阵，再聚合截面特征。
class TGC(nn.Module):
    def __init__(self, num_stocks, hidden_dim, graph_embed_dim, output_dim, embedding_init_std):
        super().__init__()
        self.M = nn.Parameter(torch.randn(num_stocks, graph_embed_dim) * embedding_init_std) #M矩阵，嵌入向量
        self.W = nn.Linear(hidden_dim, output_dim, bias=False) #W矩阵，决定聚合后的特征如何线性组合。

    def forward(self, G, stock_ids=None, *, industry_prior=None):
        if stock_ids is None:
            M = self.M
        else:
            M = self.M.index_select(0, stock_ids)

        if M.size(0) != G.size(0):
            raise ValueError("Stock embeddings must match the number of rows in G.")

        # 参考 AGCRN 的逐行归一化邻接，自环通过残差直接加入。
        logits = M @ M.T
        adjacency = torch.softmax(torch.relu(logits), dim=-1)
        output = self.W(G + adjacency @ G)
        if industry_prior is None:
            return output

        edges, valid_pairs = industry_prior
        if edges.shape != logits.shape or valid_pairs.shape != logits.shape:
            raise ValueError("Industry prior must match the current stock cross-section.")
        valid_pairs = valid_pairs.to(device=logits.device, dtype=torch.bool)
        targets = edges.to(device=logits.device, dtype=logits.dtype)
        # Use raw inner products so that negative pairs can have negative logits.
        loss = F.binary_cross_entropy_with_logits(
            logits[valid_pairs], targets[valid_pairs], reduction="sum"
        ) / valid_pairs.sum().clamp_min(1)
        return output, loss
