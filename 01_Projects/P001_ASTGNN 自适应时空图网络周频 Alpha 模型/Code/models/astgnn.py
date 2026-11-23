from torch import nn

from config import ModelConfig
from layers.residual import FeatureExtractor
from modules.gru import TemporalGRU
from modules.predictor import FactorPredictor
from modules.tgc import TGC


# 组装可选 ResNet、GRU、图卷积和预测层。
class ASTGNN(nn.Module):
    def __init__(self, num_stocks, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.feature_extractor = None
        gru_input_dim = self.config.num_features
        if self.config.use_resnet:
            self.feature_extractor = FeatureExtractor(
                input_channels=self.config.num_features,
                feature_dim=self.config.feature_dim,
                kernel_size=self.config.resnet_kernel_size,
                padding=self.config.resnet_padding,
            )
            gru_input_dim = self.config.feature_dim
        self.gru = TemporalGRU(
            input_dim=gru_input_dim,
            hidden_dim=self.config.gru_hidden_dim,
        )
        self.tgc = TGC(
            num_stocks=num_stocks,
            hidden_dim=self.config.gru_hidden_dim,
            graph_embed_dim=self.config.graph_embed_dim,
            output_dim=self.config.tgc_hidden_dim,
            embedding_init_std=self.config.tgc_embedding_init_std,
        )
        self.predictor = FactorPredictor(
            input_dim=self.config.tgc_hidden_dim,
            num_factors=self.config.num_factors,
            batch_norm_eps=self.config.batch_norm_eps,
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_affine=self.config.batch_norm_affine,
            batch_norm_track_running_stats=self.config.batch_norm_track_running_stats,
        )

    # 按配置选择输入路径，并返回中间结果和预测 C。
    def forward(self, x, stock_ids=None, *, industry_prior=None):
        if self.feature_extractor is None:
            if x.ndim != 3 or x.size(-1) != self.config.num_features:
                raise ValueError("Direct GRU input must have shape [stocks, steps, features].")
            X = x
        else:
            X = self.feature_extractor(x)
        G = self.gru(X)
        tgc_result = self.tgc(G, stock_ids, industry_prior=industry_prior)
        if industry_prior is None:
            T = tgc_result
        else:
            T, graph_loss = tgc_result
        Z, C = self.predictor(T)
        result = {"X": X, "G": G, "T": T, "Z": Z, "C": C}
        if industry_prior is not None:
            result["L_GP"] = graph_loss
        return result
