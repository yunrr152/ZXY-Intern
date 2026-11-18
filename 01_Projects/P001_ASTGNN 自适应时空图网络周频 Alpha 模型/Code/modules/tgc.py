import torch
from torch import nn


# 从股票节点嵌入学习邻接矩阵，再聚合截面特征。
class TGC(nn.Module):
    def __init__(self, num_stocks, hidden_dim, graph_embed_dim, output_dim, embedding_init_std):
        super().__init__()
        self.M = nn.Parameter(torch.randn(num_stocks, graph_embed_dim) * embedding_init_std) #M矩阵，嵌入向量
        self.W = nn.Linear(hidden_dim, output_dim, bias=False) #W矩阵，决定聚合后的特征如何线性组合。

    def forward(self, G, stock_ids=None):
        if stock_ids is None:
            M = self.M
        else:
            M = self.M.index_select(0, stock_ids)

        if M.size(0) != G.size(0):
            raise ValueError("Stock embeddings must match the number of rows in G.")

        # 参考 AGCRN 的逐行归一化邻接，自环通过残差直接加入。
        adjacency = torch.softmax(torch.relu(M @ M.T), dim=-1)
        return self.W(G + adjacency @ G)
