from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        return self.relu(self.conv2(out) + x)


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=6, feature_dim=64):
        super().__init__()
        self.input_conv = nn.Conv1d(input_channels, feature_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.residual = ResidualBlock(feature_dim)

    def forward(self, x):
        num_stocks, seq_len, num_days, num_features = x.shape
        slices = x.reshape(num_stocks * seq_len, num_days, num_features)
        features = self.relu(self.input_conv(slices.transpose(1, 2)))
        features = self.residual(features).mean(dim=-1)
        return features.reshape(num_stocks, seq_len, -1)
