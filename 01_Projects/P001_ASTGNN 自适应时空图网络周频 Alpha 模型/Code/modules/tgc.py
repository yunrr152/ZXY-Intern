import torch
from torch import nn


class TGC(nn.Module):
    def __init__(self, num_stocks, hidden_dim, graph_embed_dim, output_dim):
        super().__init__()
        self.M = nn.Parameter(torch.randn(num_stocks, graph_embed_dim) * 0.01)
        self.W = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, G, stock_ids=None):
        if stock_ids is None:
            M = self.M
        else:
            M = self.M.index_select(0, stock_ids)

        if M.size(0) != G.size(0):
            raise ValueError("Stock embeddings must match the number of rows in G.")

        adjacency = torch.softmax(torch.relu(M @ M.T), dim=1)
        identity = torch.eye(M.size(0), dtype=adjacency.dtype, device=adjacency.device)
        return self.W((identity + adjacency) @ G)
