from dataclasses import dataclass


# 集中配置输入维度、模型结构和训练超参数。
@dataclass
class ModelConfig:
    # 输入与模型结构
    seq_len: int = 30
    num_features: int = 6
    num_days: int = 5
    feature_dim: int = 6
    use_resnet: bool = True
    resnet_kernel_size: int = 3
    resnet_padding: int = 1
    gru_hidden_dim: int = 64
    graph_embed_dim: int = 16
    tgc_hidden_dim: int = 64
    tgc_embedding_init_std: float = 0.01
    num_factors: int = 64
    batch_norm_eps: float = 1e-5
    batch_norm_momentum: float = 0.1
    batch_norm_affine: bool = True
    batch_norm_track_running_stats: bool = True

    # 数据预处理与标签
    winsorize_mad_multiplier: float = 7.0
    target_start_offset: int = 1
    target_end_offset: int = 11

    # 年度滚动训练
    training_years: int = 10
    random_seed: int = 42
    ensemble_seeds: tuple[int, ...] = (0, 47, 427)
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_epochs: int = 1
    checkpoint_every_training_days: int = 100
    min_stocks_per_cross_section: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_amsgrad: bool = False
    epochs: int = 20
    lambda_orth: float = 0.01
    lambda_adv: float = 0.01
    # IFGSM
    adv_epsilon: float = 0.01
    adv_step_size: float = 0.005
    adv_steps: int = 3
    lgbols_lookback_years: int = 1
    lgbols_num_boost_round: int = 100
    lgbols_objective: str = "regression"
    lgbols_boosting_type: str = "gbdt"
    lgbols_learning_rate: float = 0.05
    lgbols_num_leaves: int = 31
    lgbols_num_threads: int = 1
    lgbols_verbosity: int = -1
