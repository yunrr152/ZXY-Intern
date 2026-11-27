def require_lightgbm():
    try:
        import lightgbm
    except ModuleNotFoundError as error:
        if error.name != "lightgbm":
            raise
        raise RuntimeError(
            "lgbols requires LightGBM. Install the dependencies in requirements.txt."
        ) from error
    return lightgbm


# 用冻结网络生成的弱因子拟合 LightGBM 的平方误差回归树。
def fit_lgbols(factors, targets, config):
    lightgbm = require_lightgbm()
    #转换数据格式为Numpy数组，训练数据封装Z和Y
    dataset = lightgbm.Dataset(factors.numpy(), label=targets.numpy())
    return lightgbm.train(
        {
            "boosting_type": config.lgbols_boosting_type,
            "objective": config.lgbols_objective,
            "learning_rate": config.lgbols_learning_rate,
            "num_leaves": config.lgbols_num_leaves,
            "seed": config.random_seed,
            "num_threads": config.lgbols_num_threads,
            "verbosity": config.lgbols_verbosity,
        },
        dataset,
        num_boost_round=config.lgbols_num_boost_round,
    )
