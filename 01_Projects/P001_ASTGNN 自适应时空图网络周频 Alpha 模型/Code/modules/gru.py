import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_u = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_u = nn.Parameter(torch.zeros(hidden_dim))
        self.W_r = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        self.W_g = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_g = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, g_prev):
        u = torch.sigmoid(self.W_u(x) + self.U_u(g_prev) + self.b_u)
        r = torch.sigmoid(self.W_r(x) + self.U_r(g_prev) + self.b_r)
        g_tilde = torch.tanh(self.W_g(x) + self.U_g(r * g_prev) + self.b_g)
        return (1 - u) * g_prev + u * g_tilde


class MultiSeedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, seeds=(42, 123, 2026)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            for seed in seeds:
                torch.random.default_generator.manual_seed(seed)
                self.cells.append(GRUCell(input_dim, hidden_dim))

    def forward(self, x):
        final_states = []
        for cell in self.cells:
            hidden = x.new_zeros(x.size(0), self.hidden_dim)
            for step in x.unbind(dim=1):
                hidden = cell(step, hidden)
            final_states.append(hidden)
        return torch.stack(final_states, dim=0).mean(dim=0)
