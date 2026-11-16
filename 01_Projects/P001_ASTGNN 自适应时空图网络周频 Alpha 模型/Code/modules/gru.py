import torch
from torch import nn


# 独立初始化多路 GRU，并平均各路的最后隐藏状态。
class MultiSeedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, seeds=(42, 123, 2026)):
        super().__init__()
        self.branches = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            for seed in seeds:
                torch.random.default_generator.manual_seed(seed)
                self.branches.append(nn.GRU(input_dim, hidden_dim, batch_first=True))

    def forward(self, x):
        final_states = []
        for branch in self.branches:
            _, hidden = branch(x)
            final_states.append(hidden[0])
        return torch.stack(final_states, dim=0).mean(dim=0)
