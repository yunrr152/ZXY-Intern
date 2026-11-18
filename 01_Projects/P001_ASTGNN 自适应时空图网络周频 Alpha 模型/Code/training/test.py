import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch

from config import ModelConfig
from data.day import DayPanel
from training.lgbols import require_lightgbm
from training.train import annual_scores


# 固定一个预测年度训练并输出分数，不遍历其他年度。
def main():
    parser = argparse.ArgumentParser(
        description="Train ASTGNN for one fixed day-data prediction year."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="UTF-8 CSV: date,stock_id,high,open,low,close,vwap,volume",
    )
    parser.add_argument("--prediction-year", type=int, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output CSV for this prediction year"
    )
    defaults = ModelConfig()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--lgbols-lookback-years",
        type=int,
        default=defaults.lgbols_lookback_years,
    )
    parser.add_argument(
        "--lgbols-num-boost-round",
        type=int,
        default=defaults.lgbols_num_boost_round,
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if (
        not 1 <= args.lgbols_lookback_years <= defaults.training_years
        or args.lgbols_num_boost_round < 1
    ):
        parser.error("lgbols lookback must be within the training period and boost rounds positive")

    panel = DayPanel(args.data)
    years = {current_date.year for current_date in panel.dates}
    earliest_year = panel.dates[0].year
    if args.prediction_year not in years:
        parser.error("--prediction-year has no observations in the input data")
    if args.prediction_year - earliest_year < defaults.training_years:
        parser.error(
            f"--prediction-year needs {defaults.training_years} preceding calendar years of data"
        )

    require_lightgbm()
    config = replace(
        ModelConfig(),
        use_resnet=False,
        epochs=args.epochs,
        device=args.device,
        random_seed=args.seed,
        lgbols_lookback_years=args.lgbols_lookback_years,
        lgbols_num_boost_round=args.lgbols_num_boost_round,
    )
    with args.output.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("date", "stock_id", "prediction_year", "C"))
        writer.writerows(
            annual_scores(panel, args.prediction_year, config, torch.device(config.device))
        )
    print(f"Prediction-year scores written to {args.output}")


if __name__ == "__main__":
    main()
