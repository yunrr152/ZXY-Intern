from dataclasses import dataclass


# 集中配置输入维度、模型结构和训练超参数。
@dataclass
class ModelConfig:
    seq_len: int = 30
    num_features: int = 6
    num_days: int = 5
    feature_dim: int = 6
    gru_hidden_dim: int = 64
    gru_seeds: tuple = (0, 47, 427)
    graph_embed_dim: int = 16
    tgc_hidden_dim: int = 64
    num_factors: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    lambda_orth: float = 0.01
    lambda_adv: float = 0.01
    use_resnet: bool = True
    adv_epsilon: float = 0.01
    adv_step_size: float = 0.005
    adv_steps: int = 3
    lgbols_lookback_years: int = 1
    lgbols_num_boost_round: int = 100
