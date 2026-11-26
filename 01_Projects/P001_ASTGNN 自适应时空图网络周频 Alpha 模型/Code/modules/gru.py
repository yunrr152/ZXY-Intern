from torch import nn


class TemporalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.recurrent = nn.GRU(input_dim, hidden_dim, batch_first=True) #把batch，即N放到第一个

    def forward(self, x):
        _, hidden = self.recurrent(x) #经过30次更新，保留最后一个时间步
        return hidden[0]
