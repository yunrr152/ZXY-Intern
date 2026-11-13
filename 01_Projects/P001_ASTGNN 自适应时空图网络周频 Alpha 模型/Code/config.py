from dataclasses import dataclass


@dataclass
class ModelConfig:
    seq_len: int = 30
    num_features: int = 6
    num_days: int = 5
    feature_dim: int = 64
    gru_hidden_dim: int = 64
    gru_seeds: tuple = (42, 123, 2026)
    graph_embed_dim: int = 16
    tgc_hidden_dim: int = 64
    num_factors: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    lambda_orth: float = 0.01
    lambda_adv: float = 0.1
    lambda_gp: float = 0.01
