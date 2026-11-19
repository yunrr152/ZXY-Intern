
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 1. 文件路径
# ============================================================

ROOT = Path(__file__).resolve().parent

SCORES_PATH = ROOT / "data/output/gpu_2017_scores.csv"
MASK_PATH = ROOT / "data/output/day_pool_mask_2007_2017.csv"

PREDICTION_YEAR = 2017

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.max_rows", 30)


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def main():

    # ========================================================
    # 2. 只读加载 CSV
    # ========================================================

    section("1. 读取文件")

    if not SCORES_PATH.is_file():
        raise FileNotFoundError(SCORES_PATH)

    if not MASK_PATH.is_file():
        raise FileNotFoundError(MASK_PATH)

    scores = pd.read_csv(
        SCORES_PATH,
        encoding="utf-8-sig",
        dtype={"stock_id": "string"}
    )

    mask = pd.read_csv(
        MASK_PATH,
        encoding="utf-8-sig",
        usecols=["date", "stock_id", "in_pool"],
        dtype={"stock_id": "string"}
    )

    print("Scores shape:", scores.shape)
    print("Mask shape:", mask.shape)
    print("\nScores columns:", scores.columns.tolist())
    print("\nScores 前 5 行:")
    print(scores.head())

    # ========================================================
    # 3. 字段检查
    # ========================================================

    section("2. 字段完整性")

    required_scores = {"date", "stock_id", "C"}
    required_mask = {"date", "stock_id", "in_pool"}

    missing_scores = required_scores - set(scores.columns)
    missing_mask = required_mask - set(mask.columns)

    if missing_scores:
        raise ValueError(
            f"Scores 缺少字段: {sorted(missing_scores)}"
        )

    if missing_mask:
        raise ValueError(
            f"Mask 缺少字段: {sorted(missing_mask)}"
        )

    # 日期转换
    scores["date"] = pd.to_datetime(
        scores["date"], errors="raise"
    )
    mask["date"] = pd.to_datetime(
        mask["date"], errors="raise"
    )

    # 统一股票代码表示
    scores["stock_id"] = scores["stock_id"].str.strip()
    mask["stock_id"] = mask["stock_id"].str.strip()

    print("必要字段检查通过")

    # ========================================================
    # 4. 日期范围
    # ========================================================

    section("3. 日期范围检查")

    print("最早日期:", scores["date"].min())
    print("最晚日期:", scores["date"].max())
    print("交易日数量:", scores["date"].nunique())

    years = scores["date"].dt.year.value_counts().sort_index()

    print("\n各年份记录数:")
    print(years.to_string())

    wrong_year = scores[
        scores["date"].dt.year != PREDICTION_YEAR
    ]

    print(
        f"\n非 {PREDICTION_YEAR} 年记录数:",
        len(wrong_year)
    )

    # ========================================================
    # 5. 股票数量与每日截面
    # ========================================================

    section("4. 股票数量与每日截面")

    print(
        "全期间不同股票数量:",
        scores["stock_id"].nunique()
    )

    daily_counts = (
        scores.groupby("date")
        .agg(
            rows=("stock_id", "size"),
            unique_stocks=("stock_id", "nunique")
        )
    )

    print("\n每日截面数量统计:")
    print(daily_counts.describe().to_string())

    print("\n截面数量最少的 10 天:")
    print(daily_counts.nsmallest(10, "rows").to_string())

    print("\n截面数量最多的 10 天:")
    print(daily_counts.nlargest(10, "rows").to_string())

    print("\n前 10 个交易日:")
    print(daily_counts.head(10).to_string())

    print("\n最后 10 个交易日:")
    print(daily_counts.tail(10).to_string())

    # ========================================================
    # 6. 重复键
    # ========================================================

    section("5. 重复键检查")

    duplicate_mask = scores.duplicated(
        subset=["date", "stock_id"],
        keep=False
    )

    duplicates = scores.loc[duplicate_mask]

    print("重复 (date, stock_id) 记录数:", len(duplicates))
    print(
        "重复键组数:",
        duplicates[["date", "stock_id"]]
        .drop_duplicates()
        .shape[0]
    )

    if not duplicates.empty:
        print("\n重复记录示例:")
        print(duplicates.head(20).to_string(index=False))

    # ========================================================
    # 7. C 数值质量
    # ========================================================

    section("6. C 的 NaN / Inf 检查")

    C = pd.to_numeric(scores["C"], errors="coerce")

    values = C.to_numpy(dtype="float64")

    nan_count = np.isnan(values).sum()
    pos_inf_count = np.isposinf(values).sum()
    neg_inf_count = np.isneginf(values).sum()

    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]

    print("总记录数:", len(C))
    print("NaN 数量:", nan_count)
    print("+Inf 数量:", pos_inf_count)
    print("-Inf 数量:", neg_inf_count)
    print("有限值数量:", finite_mask.sum())

    print(
        "非有限值占比:",
        f"{(~finite_mask).mean():.4%}"
    )

    # ========================================================
    # 8. C 的分布统计
    # ========================================================

    section("7. C 的均值 / 标准差 / 分位数")

    if len(finite_values) > 0:

        s = pd.Series(finite_values)

        print("均值:", s.mean())
        print("标准差:", s.std())
        print("最小值:", s.min())
        print("最大值:", s.max())

        quantiles = s.quantile([
            0.00,
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00
        ])

        print("\n分位数:")
        print(quantiles.to_string())

    else:
        print("没有有限的 C 数值")

    # ========================================================
    # 9. 每日 C 分布
    # ========================================================

    section("8. 每日 C 分布")

    scores["C_numeric"] = C.replace(
        [np.inf, -np.inf], np.nan
    )

    daily_C = (
        scores.groupby("date")["C_numeric"]
        .agg(["count", "mean", "std", "min", "max"])
    )

    print("\n每日统计量的整体分布:")
    print(daily_C.describe().to_string())

    print("\n前 5 日:")
    print(daily_C.head().to_string())

    print("\n后 5 日:")
    print(daily_C.tail().to_string())

    # ========================================================
    # 10. 检查股票池 Mask
    # ========================================================

    section("9. 股票池一致性验证")

    mask_duplicates = mask.duplicated(
        ["date", "stock_id"],
        keep=False
    )

    print(
        "Mask 重复键记录数:",
        mask_duplicates.sum()
    )

    mask["in_pool"] = pd.to_numeric(
        mask["in_pool"], errors="coerce"
    )

    invalid_mask = ~mask["in_pool"].isin([0, 1])

    print(
        "Mask 非法 in_pool 值数量:",
        invalid_mask.sum()
    )

    # 不允许 Mask 存在重复键，否则 merge 可能放大记录数
    if mask_duplicates.any():
        raise ValueError(
            "Mask 存在重复 (date, stock_id)，无法可靠验证"
        )

    # 左连接：保留全部预测结果
    merged = scores.merge(
        mask[["date", "stock_id", "in_pool"]],
        on=["date", "stock_id"],
        how="left",
        validate="many_to_one",
        indicator=True
    )

    unmatched = merged["_merge"] != "both"

    out_of_pool = (
        (merged["_merge"] == "both")
        & (merged["in_pool"] == 0)
    )

    valid_pool = (
        (merged["_merge"] == "both")
        & (merged["in_pool"] == 1)
    )

    invalid_pool = (
        (merged["_merge"] == "both")
        & ~merged["in_pool"].isin([0, 1])
    )

    print("预测记录总数:", len(merged))
    print("匹配到 Mask 的记录数:", (~unmatched).sum())
    print("未匹配到 Mask 的记录数:", unmatched.sum())
    print("in_pool == 1 的预测记录数:", valid_pool.sum())
    print("in_pool == 0 的预测记录数:", out_of_pool.sum())
    print("匹配但 Mask 值非法的记录数:", invalid_pool.sum())

    pool_pass = (
        not unmatched.any()
        and not out_of_pool.any()
        and not invalid_pool.any()
    )

    print(
        "\n是否所有预测记录均属于当日股票池:",
        "PASS" if pool_pass else "FAIL"
    )

    if unmatched.any():
        print("\n未匹配 Mask 的示例:")
        print(
            merged.loc[
                unmatched, ["date", "stock_id"]
            ].head(10).to_string(index=False)
        )

    if out_of_pool.any():
        print("\n池外股票的预测记录示例:")
        print(
            merged.loc[
                out_of_pool,
                ["date", "stock_id", "C_numeric", "in_pool"]
            ].head(10).to_string(index=False)
        )

    # ========================================================
    # 11. 最终汇总
    # ========================================================

    section("10. 最终验证汇总")

    checks = {
        "预测年份正确": len(wrong_year) == 0,
        "无重复预测键": len(duplicates) == 0,
        "C 无 NaN / Inf": finite_mask.all(),
        "所有预测股票均属于当日股票池": pool_pass,
    }

    for name, passed in checks.items():
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    print("\n验证完成，原始 CSV 未被修改。")


if __name__ == "__main__":
    main()
