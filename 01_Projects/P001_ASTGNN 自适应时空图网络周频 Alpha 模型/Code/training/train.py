import argparse
from bisect import bisect_left
import csv
from dataclasses import replace
from datetime import date
from pathlib import Path

import torch

from config import ModelConfig
from data.day import DayPanel
from data.preprocessing import FeaturePreprocessor
from models.astgnn import ASTGNN
from ops.ifgsm import ifgsm, suspend_running_statistics
from training.lgbols import fit_lgbols, require_lightgbm
from training.loss import orthogonality_loss, prediction_loss


# 以同一个模型计算干净样本、正交惩罚和对抗样本损失。
def training_loss(model, x, stock_ids, target, valid, config):
    clean = model(x, stock_ids)
    loss = prediction_loss(clean["C"], target, valid)
    loss = loss + config.lambda_orth * orthogonality_loss(clean["Z"])
    if config.lambda_adv:
        def objective(inputs):
            return prediction_loss(model(inputs, stock_ids)["C"], target, valid)

        with suspend_running_statistics(model):
            adversarial = ifgsm(
                x, objective, config.adv_epsilon, config.adv_step_size, config.adv_steps
            )
            loss = loss + config.lambda_adv * objective(adversarial)
    return loss


# 单轮滚动训练，并逐日输出该验证年的预测分数。
def annual_scores(panel, year, config, device):
    start = bisect_left(panel.dates, date(year - config.training_years, 1, 1))
    stop = bisect_left(panel.dates, date(year, 1, 1))
    end = bisect_left(panel.dates, date(year + 1, 1, 1))
    if stop == end:
        return

    preprocessor = FeaturePreprocessor(config.winsorize_mad_multiplier).fit(
        torch.cat(panel.daily[start:stop])
    )
    active_ids = torch.unique(torch.cat(panel.stock_rows[start:stop]))
    unknown = active_ids.numel()
    id_map = torch.full((len(panel.stocks),), unknown, dtype=torch.long)
    id_map[active_ids] = torch.arange(unknown)

    #初始化ADTGNN
    torch.manual_seed(config.random_seed)
    model = ASTGNN(unknown + 1, config).to(device)
    with torch.no_grad():
        model.tgc.M[unknown].zero_() #未知股票节点的初始嵌入设为全零
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        amsgrad=config.adam_amsgrad,
    )
    # 标签终点必须早于验证年。
    training_days = range(
        max(start, config.seq_len - 1),
        stop - config.target_end_offset,
    )
    for epoch in range(config.epochs):
        model.train()
        steps = 0
        total_loss = 0.0
        for day in training_days:
            ids = panel.stock_rows[day]
            target = panel.target(
                day, ids, config.target_start_offset, config.target_end_offset
            )
            valid = torch.isfinite(target) 
            
            #模型不会在股票数量过少或完全没有有效监督标签的日期进行训练。
            if ids.numel() < config.min_stocks_per_cross_section or not valid.any():
                continue
            _, x = panel.window(day, config.seq_len)
            x = preprocessor.transform(x).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(
                model, x, id_map[ids].to(device), target.to(device), valid.to(device), config
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss on {panel.dates[day]}")
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            steps += 1
        if not steps:
            raise ValueError(f"No usable training cross-sections for validation year {year}.")
        print(f"{year} epoch {epoch + 1}/{config.epochs}: loss={total_loss / steps:.6g}")

    model.eval()
    # 冻结模型，用验证年前的近期样本训练树，标签沿用阶段一的时间边界。
    tree_start = bisect_left(panel.dates, date(year - config.lgbols_lookback_years, 1, 1))
    tree_factors, tree_targets = [], []
    with torch.no_grad():
        for day in range(max(training_days.start, tree_start), training_days.stop):
            ids, x = panel.window(day, config.seq_len)
            target = panel.target(
                day, ids, config.target_start_offset, config.target_end_offset
            )
            valid = torch.isfinite(target)
            if not valid.any():
                continue
            x = preprocessor.transform(x).to(device)
            factors = model(x, id_map[ids].to(device))["Z"].cpu()
            tree_factors.append(factors[valid])
            tree_targets.append(target[valid])
    if not tree_targets:
        raise ValueError(f"No labeled lgbols samples before validation year {year}.")
    booster = fit_lgbols(torch.cat(tree_factors), torch.cat(tree_targets), config)
    for day in range(stop, end):
        with torch.no_grad():
            ids, x = panel.window(day, config.seq_len)
            x = preprocessor.transform(x).to(device)
            factors = model(x, id_map[ids].to(device))["Z"].cpu()
            scores = torch.from_numpy(
                booster.predict(factors.numpy(), num_threads=config.lgbols_num_threads)
            )
            if not torch.isfinite(scores).all():
                raise RuntimeError(f"Non-finite predictions on {panel.dates[day]}")
        for stock_id, score in zip(ids.tolist(), scores.tolist()):
            yield panel.dates[day].isoformat(), panel.stocks[stock_id], year, score


# 读取命令行参数，按验证年份汇总预测结果。
def main():
    parser = argparse.ArgumentParser(description="Train ASTGNN on rolling day panels.")
    parser.add_argument("--data", type=Path, required=True, help="UTF-8 CSV: date,stock_id,high,open,low,close,vwap,volume")
    parser.add_argument("--output", type=Path, required=True, help="New output CSV for all validation scores")
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
    require_lightgbm()

    panel = DayPanel(args.data)
    first_year = panel.dates[0].year + defaults.training_years
    last_year = panel.dates[-1].year
    if first_year > last_year:
        parser.error(
            f"The dataset needs {defaults.training_years} training calendar years "
            "and a validation year."
        )
    config = replace(
        ModelConfig(),
        use_resnet=False,
        epochs=args.epochs,
        device=args.device,
        random_seed=args.seed,
        lgbols_lookback_years=args.lgbols_lookback_years,
        lgbols_num_boost_round=args.lgbols_num_boost_round,
    )
    device = torch.device(config.device)
    with args.output.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("date", "stock_id", "validation_year", "C"))
        for year in range(first_year, last_year + 1):
            writer.writerows(annual_scores(panel, year, config, device))
    print(f"Validation scores written to {args.output}")


if __name__ == "__main__":
    main()
