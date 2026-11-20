from torch import nn


class TemporalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.recurrent = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, hidden = self.recurrent(x)
        return hidden[0]
