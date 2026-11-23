import argparse
import csv
from dataclasses import replace
import os
from pathlib import Path
from time import perf_counter

import torch

from config import ModelConfig
from dataset.day import DayPanel
from training.lgbols import require_lightgbm
from training.train import (
    add_industry_prior_arguments, annual_scores, configure_logger, industry_prior_options,
    report, resolve_device, stage,
)


# 固定一个预测年度训练并输出分数，不遍历其他年度。
def main():
    run_started = perf_counter()
    parser = argparse.ArgumentParser(
        description="Train ASTGNN for one fixed day-data prediction year."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="UTF-8 CSV: date,stock_id,high,open,low,close,vwap,volume",
    )
    parser.add_argument("--mask", type=Path, required=True, help="UTF-8 CSV: date,stock_id,in_pool")
    parser.add_argument("--prediction-year", type=int, required=True)
    parser.add_argument(
        "--train-start-year",
        type=int,
        help="Optional explicit first training calendar year; must match training_years.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output CSV for this prediction year"
    )
    defaults = ModelConfig()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(defaults.checkpoint_dir))
    parser.add_argument("--resume", action="store_true", help="Resume the model from its epoch checkpoint.")
    parser.add_argument("--log-file", type=Path, help="Optional UTF-8 run log path.")
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
    add_industry_prior_arguments(parser, defaults)
    args = parser.parse_args()
    industry_prior_options(parser, args)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if defaults.checkpoint_every_epochs < 1:
        parser.error("config.checkpoint_every_epochs must be positive")
    if defaults.checkpoint_every_training_days < 1:
        parser.error("config.checkpoint_every_training_days must be positive")
    if (
        not 1 <= args.lgbols_lookback_years <= defaults.training_years
        or args.lgbols_num_boost_round < 1
    ):
        parser.error("lgbols lookback must be within the training period and boost rounds positive")

    log_path = args.log_file or args.output.with_suffix(".log")
    logger = configure_logger(log_path)
    report(f"Run log: {log_path}", logger)
    stage(1, 6, "Loading day feature panel...", logger)
    panel = DayPanel(args.data, args.mask, logger)
    years = {current_date.year for current_date in panel.dates}
    train_start_year = (
        args.train_start_year
        if args.train_start_year is not None
        else args.prediction_year - defaults.training_years
    )
    if args.prediction_year not in years:
        parser.error("--prediction-year has no observations in the input data")
    if args.prediction_year - train_start_year != defaults.training_years:
        parser.error(
            "--prediction-year minus --train-start-year must equal "
            "config.training_years"
        )
    if panel.dates[0].year > train_start_year:
        parser.error(
            f"Input data begins in {panel.dates[0].year}, after requested training start "
            f"year {train_start_year}"
        )

    require_lightgbm()
    report(
        f"Training from {train_start_year} and predicting {args.prediction_year}; "
        "this may take some time.",
        logger,
    )
    config = replace(
        ModelConfig(),
        use_resnet=False,
        epochs=args.epochs,
        device=args.device,
        random_seed=args.seed,
        checkpoint_dir=str(args.checkpoint_dir),
        lgbols_lookback_years=args.lgbols_lookback_years,
        lgbols_num_boost_round=args.lgbols_num_boost_round,
    )
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("date", "stock_id", "prediction_year", "C"))
        writer.writerows(
            annual_scores(
                panel,
                args.prediction_year,
                config,
                resolve_device(config.device, logger),
                args.checkpoint_dir / f"astgnn_{args.prediction_year}.pt",
                args.resume,
                logger,
                industry_prior_dir=args.industry_prior_dir,
                lambda_gp=args.lambda_gp,
            )
        )
    os.replace(temporary_output, args.output)
    report(f"Prediction-year scores written to {args.output}", logger)
    report(f"Fixed-year test completed in {perf_counter() - run_started:.1f}s.", logger)


if __name__ == "__main__":
    main()
