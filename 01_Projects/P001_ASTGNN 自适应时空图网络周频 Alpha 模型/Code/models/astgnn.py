from torch import nn

from config import ModelConfig
from layers.residual import FeatureExtractor
from modules.gru import MultiSeedGRU
from modules.predictor import FactorPredictor
from modules.tgc import TGC


class ASTGNN(nn.Module):
    def __init__(self, num_stocks, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.feature_extractor = FeatureExtractor(
            input_channels=self.config.num_features,
            feature_dim=self.config.feature_dim,
        )
        self.gru = MultiSeedGRU(
            input_dim=self.config.feature_dim,
            hidden_dim=self.config.gru_hidden_dim,
            seeds=self.config.gru_seeds,
        )
        self.tgc = TGC(
            num_stocks=num_stocks,
            hidden_dim=self.config.gru_hidden_dim,
            graph_embed_dim=self.config.graph_embed_dim,
            output_dim=self.config.tgc_hidden_dim,
        )
        self.predictor = FactorPredictor(
            input_dim=self.config.tgc_hidden_dim,
            num_factors=self.config.num_factors,
        )

    def forward(self, x, stock_ids=None):
        X = self.feature_extractor(x)
        G = self.gru(X)
        T = self.tgc(G, stock_ids)
        Z, C = self.predictor(T)
        return {"X": X, "G": G, "T": T, "Z": Z, "C": C}
