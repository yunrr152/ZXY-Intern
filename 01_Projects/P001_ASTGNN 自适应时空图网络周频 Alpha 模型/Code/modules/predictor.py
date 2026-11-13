from torch import nn


class FactorPredictor(nn.Module):
    def __init__(self, input_dim=64, num_factors=64):
        super().__init__()
        self.num_factors = num_factors
        self.linear = nn.Linear(input_dim, num_factors)
        self.bn_factors = nn.BatchNorm1d(num_factors)
        self.bn_score = nn.BatchNorm1d(1)

    def forward(self, T):
        Z = self.bn_factors(self.linear(T))
        C = self.bn_score(Z.mean(dim=1, keepdim=True))
        return Z, C.squeeze(-1)
