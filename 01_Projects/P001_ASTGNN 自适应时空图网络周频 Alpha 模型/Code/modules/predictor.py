from torch import nn


# 生成弱因子 Z，等权分数用于神经网络的可微训练。
class FactorPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        num_factors,
        batch_norm_eps,
        batch_norm_momentum,
        batch_norm_affine,
        batch_norm_track_running_stats,
    ):
        super().__init__()
        self.num_factors = num_factors
        self.linear = nn.Linear(input_dim, num_factors)
        batch_norm = {
            "eps": batch_norm_eps,
            "momentum": batch_norm_momentum,
            "affine": batch_norm_affine,
            "track_running_stats": batch_norm_track_running_stats,
        }
        self.bn_factors = nn.BatchNorm1d(num_factors, **batch_norm)
        self.bn_score = nn.BatchNorm1d(1, **batch_norm)

    def forward(self, T):
        Z = self.bn_factors(self.linear(T))
        C = self.bn_score(Z.mean(dim=1, keepdim=True))
        return Z, C.squeeze(-1) #C的第二个维度为1的时候才能去除，和标签Y对齐
