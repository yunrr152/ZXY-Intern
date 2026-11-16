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
def fit_lgbols(factors, targets, num_boost_round, seed):
    lightgbm = require_lightgbm()
    dataset = lightgbm.Dataset(factors.numpy(), label=targets.numpy())
    return lightgbm.train(
        {
            "boosting_type": "gbdt",
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "seed": seed,
            "num_threads": 1,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=num_boost_round,
    )
