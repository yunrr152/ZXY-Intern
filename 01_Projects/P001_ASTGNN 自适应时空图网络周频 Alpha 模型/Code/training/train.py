import argparse
from bisect import bisect_left
import csv
from dataclasses import replace
from datetime import date, datetime
from itertools import islice
import logging
import math
import os
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm

from config import ModelConfig
from dataset.day import DayPanel
from dataset.industry_relation import SavedIndustryPrior
from dataset.preprocessing import FeaturePreprocessor
from models.astgnn import ASTGNN
from ops.ifgsm import ifgsm, suspend_running_statistics
from training.lgbols import fit_lgbols, require_lightgbm
from training.loss import orthogonality_loss, prediction_loss

#日志记录
def configure_logger(log_path):
    """Write persistent run messages to a user-visible UTF-8 log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("astgnn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def report(message, logger=None):
    tqdm.write(message)
    if logger is not None:
        logger.info(message)


def stage(index, total, message, logger=None):
    report(f"[{index}/{total}] {message}", logger)

#行业先验矩阵
def add_industry_prior_arguments(parser, defaults):
    parser.add_argument("--industry-prior-dir", type=Path, default=defaults.industry_prior_dir,
                        help="Offline industry matrix export.")
    parser.add_argument("--lambda-gp", type=float, default=defaults.lambda_gp,
                        help="Industry graph loss weight.")


def industry_prior_options(parser, args):
    if not math.isfinite(args.lambda_gp) or args.lambda_gp < 0:
        parser.error("--lambda-gp must be finite and nonnegative.")
    if args.lambda_gp > 0:
        if args.industry_prior_dir is None:
            parser.error("--lambda-gp requires --industry-prior-dir.")
        args.industry_prior_dir = args.industry_prior_dir.resolve()
        try:
            SavedIndustryPrior(args.industry_prior_dir)
        except ValueError as error:
            parser.error(str(error))

#设备管理
def resolve_device(device_name, logger=None):
    """Validate the requested compute device before allocating model tensors."""
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. "
                f"PyTorch={torch.__version__}, PyTorch CUDA build={torch.version.cuda}."
            )
        report(f"Using GPU: {torch.cuda.get_device_name(device)}", logger)
    else:
        report(f"Using device: {device}", logger)
    return device

#模型保护与恢复
def save_checkpoint(
    path,
    year,
    completed_epochs,
    in_progress_epoch,
    next_day_offset,
    running_loss,
    steps,
    skipped,
    model,
    optimizer,
    preprocessor,
    active_ids,
    logger,
    gp_settings=None,
):
    """Atomically save the state needed to resume at an epoch or training-day boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    statistics = tuple(value.detach().cpu() for value in preprocessor._statistics)
    torch.save(
        {
            "year": year,
            "completed_epochs": completed_epochs,
            "in_progress_epoch": in_progress_epoch,
            "next_day_offset": next_day_offset,
            "running_loss": running_loss,
            "steps": steps,
            "skipped": skipped,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "preprocessor_statistics": statistics,
            "active_ids": active_ids.cpu(),
            "gp_settings": gp_settings,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)
    if in_progress_epoch is None:
        report(f"{year}: checkpoint saved after epoch {completed_epochs} to {path}.", logger)
    else:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        report(
            f"{year}: heartbeat {timestamp}; epoch {in_progress_epoch}, "
            f"next training-day offset {next_day_offset}, checkpoint saved to {path}.",
            logger,
        )

#恢复函数
def restore_checkpoint(path, year, model, optimizer, preprocessor, active_ids, device, logger,
                       gp_settings=None):
    """Restore a same-year checkpoint and verify that its stock-node universe matches."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("gp_settings") != gp_settings:
        raise ValueError("Checkpoint industry prior or graph loss weight differs from this run.")
    if checkpoint.get("year") != year:
        raise ValueError(f"Checkpoint {path} belongs to year {checkpoint.get('year')}, not {year}.")
    if not torch.equal(checkpoint["active_ids"].cpu(), active_ids.cpu()):
        raise ValueError("Checkpoint stock-node universe differs from the current training data.")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    preprocessor._statistics = tuple(checkpoint["preprocessor_statistics"])
    # Support checkpoints saved before intra-epoch heartbeat support was added.
    completed_epochs = int(checkpoint.get("completed_epochs", checkpoint.get("epoch_completed", 0)))
    state = {
        "completed_epochs": completed_epochs,
        "in_progress_epoch": checkpoint.get("in_progress_epoch"),
        "next_day_offset": int(checkpoint.get("next_day_offset", 0)),
        "running_loss": float(checkpoint.get("running_loss", 0.0)),
        "steps": int(checkpoint.get("steps", 0)),
        "skipped": int(checkpoint.get("skipped", 0)),
    }
    if state["in_progress_epoch"] is None:
        report(f"{year}: resumed checkpoint after epoch {completed_epochs} from {path}.", logger)
    else:
        report(
            f"{year}: resumed epoch {state['in_progress_epoch']} from training-day offset "
            f"{state['next_day_offset']} using {path}.",
            logger,
        )
    return state


# 以同一个模型计算干净样本、正交惩罚和对抗样本损失。
def training_loss(model, x, stock_ids, target, valid, config, *, industry_prior=None, lambda_gp=0.0):
    clean = model(x, stock_ids, industry_prior=industry_prior)
    loss = prediction_loss(clean["C"], target, valid) #预测损失
    loss = loss + config.lambda_orth * orthogonality_loss(clean["Z"]) #加入正交损失
    if industry_prior is not None: #行业图损失
        loss = loss + lambda_gp * clean["L_GP"]
    if config.lambda_adv: #对抗损失，不考虑正交损失和图损失
        def objective(inputs):
            return prediction_loss(model(inputs, stock_ids)["C"], target, valid)

        #对抗样本的统计参数不会更新统计量
        with suspend_running_statistics(model):
            adversarial = ifgsm(
                x, objective, config.adv_epsilon, config.adv_step_size, config.adv_steps
            )
            loss = loss + config.lambda_adv * objective(adversarial)
    return loss


# 单轮滚动训练，并逐日输出该验证年的预测分数。
def annual_scores(panel, year, config, device, checkpoint_path, resume, logger=None, *,
                  industry_prior_dir=None, lambda_gp=None):
    year_started = perf_counter()
    if lambda_gp is None:
        lambda_gp = config.lambda_gp
    if industry_prior_dir is None:
        industry_prior_dir = config.industry_prior_dir
    prior_reader = SavedIndustryPrior(industry_prior_dir) if lambda_gp else None
    gp_settings = (str(prior_reader.directory.resolve()), lambda_gp) if prior_reader else None
    #起始日期
    start = bisect_left(panel.dates, date(year - config.training_years, 1, 1))
    stop = bisect_left(panel.dates, date(year, 1, 1))
    end = bisect_left(panel.dates, date(year + 1, 1, 1))
    if stop == end:
        return

    stage(2, 6, f"{year}: Building training samples and preprocessing statistics...", logger)
    preprocessing_started = perf_counter()
    training_pool_features = [panel.pool_features(day) for day in range(start, stop)]
    training_pool_features = [features for features in training_pool_features if features.numel()]
    if not training_pool_features:
        raise ValueError(f"No in-pool observations before validation year {year}.")
    
    #数据预处理
    preprocessor = FeaturePreprocessor(config.winsorize_mad_multiplier).fit(
        torch.cat(training_pool_features)
    )
    report(f"{year}: preprocessing statistics fitted in {perf_counter() - preprocessing_started:.1f}s.", logger)
    
    active_pool_rows = [panel.pool_rows(day) for day in range(start, stop)] #股票索引
    active_pool_rows = [ids for ids in active_pool_rows if ids.numel()]
    if not active_pool_rows:
        raise ValueError(f"No in-pool stocks before validation year {year}.")
    active_ids = torch.unique(torch.cat(active_pool_rows))
    unknown = active_ids.numel() #unknown是训练中出现的有效节点数
    id_map = torch.full((len(panel.stocks),), unknown, dtype=torch.long)
    id_map[active_ids] = torch.arange(unknown)

    #初始化ADTGNN
    torch.manual_seed(config.random_seed)
    model = ASTGNN(unknown + 1, config).to(device) #未知股票节点映射到unknown+1
    with torch.no_grad():
        model.tgc.M[unknown].zero_() #未知股票节点的初始嵌入设为全零
    #AdamW优化器，权重衰减+梯度更新
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        amsgrad=config.adam_amsgrad,
    )
    resume_state = { #断点机制
        "completed_epochs": 0,
        "in_progress_epoch": None,
        "next_day_offset": 0,
        "running_loss": 0.0,
        "steps": 0,
        "skipped": 0,
    }
    if resume and checkpoint_path.is_file():
        resume_state = restore_checkpoint(
            checkpoint_path, year, model, optimizer, preprocessor, active_ids, device, logger,
            gp_settings,
        )
        if resume_state["completed_epochs"] > config.epochs:
            raise ValueError(
                f"Checkpoint completed {resume_state['completed_epochs']} epochs, exceeding configured "
                f"epochs={config.epochs}."
            )
    # 标签终点必须早于验证年。
    training_days = range(
        max(start, config.seq_len - 1),
        stop - config.target_end_offset, #标签的终点不会超过验证年份的起点
    )
    if prior_reader is not None and training_days and resume_state["completed_epochs"] < config.epochs:
        first_day = panel.dates[training_days.start]
        last_day = panel.dates[training_days.stop - 1]
        if prior_reader.snapshots is not None and (
            first_day < prior_reader.start or last_day > prior_reader.end
        ):
            raise ValueError(
                f"{year}: industry prior covers {prior_reader.start} through {prior_reader.end}, "
                f"but training needs {first_day} through {last_day}."
            )
        report(
            f"{year}: industry prior enabled (lambda_gp={lambda_gp:g}) "
            f"from {prior_reader.directory} for {first_day} through {last_day}.",
            logger,
        )
    stage(3, 6, f"{year}: Training ASTGNN...", logger)

    #epoh：每个交易日进行独立参数更新，整个训练区间遍历完之后为一个epoh
    for epoch in range(resume_state["completed_epochs"], config.epochs):
        epoch_started = perf_counter()
        model.train()
        #恢复当前epoh的训练状态
        resume_epoch = resume_state["in_progress_epoch"] == epoch + 1
        start_offset = resume_state["next_day_offset"] if resume_epoch else 0
        steps = resume_state["steps"] if resume_epoch else 0
        skipped = resume_state["skipped"] if resume_epoch else 0
        total_loss = resume_state["running_loss"] if resume_epoch else 0.0
        days_to_run = range(training_days.start + start_offset, training_days.stop)#得到当前epoh的训练周期
        with tqdm(
            days_to_run,
            desc=f"{year} epoch {epoch + 1}/{config.epochs}",
            unit="day",
            leave=False,
            disable=None,
            dynamic_ncols=True,
            mininterval=1.0,
            total=len(training_days),
            initial=start_offset,
        ) as progress:
            #按交易日训练
            for day_offset, day in enumerate(progress, start=start_offset):
                ids = panel.pool_rows(day)
                target = panel.target(
                    day, ids, config.target_start_offset, config.target_end_offset
                )
                valid = torch.isfinite(target)

                #模型不会在股票数量过少或完全没有有效监督标签的日期进行训练。
                if ids.numel() < config.min_stocks_per_cross_section or not valid.any():
                    skipped += 1
                else:#构建输入特征
                    _, x = panel.window(day, config.seq_len, ids)
                    x = preprocessor.transform(x).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    industry_prior = None
                    if prior_reader is not None:
                        stock_codes = [panel.stocks[i] for i in ids.tolist()]
                        industry_prior = prior_reader.for_day(panel.dates[day], stock_codes)
                    #损失函数
                    loss = training_loss(
                        model, x, id_map[ids].to(device), target.to(device), valid.to(device), config,
                        industry_prior=industry_prior, lambda_gp=lambda_gp,
                    )
                    #反向传播和梯度更新
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss on {panel.dates[day]}")
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach().item()
                    steps += 1
                if steps:
                    progress.set_postfix(
                        loss=f"{total_loss / steps:.5g}", usable=steps, skipped=skipped,
                        refresh=False,
                    )
                #定期保存checkpoint
                next_day_offset = day_offset + 1
                if (
                    next_day_offset % config.checkpoint_every_training_days == 0
                    and next_day_offset < len(training_days)
                ):
                    save_checkpoint(
                        checkpoint_path,
                        year,
                        epoch,
                        epoch + 1,
                        next_day_offset,
                        total_loss,
                        steps,
                        skipped,
                        model,
                        optimizer,
                        preprocessor,
                        active_ids,
                        logger,
                        gp_settings,
                    )
        if not steps:
            raise ValueError(f"No usable training cross-sections for validation year {year}.")
        report(
            f"{year} epoch {epoch + 1}/{config.epochs}: loss={total_loss / steps:.6g}, "
            f"time={perf_counter() - epoch_started:.1f}s",
            logger,
        )
        if (epoch + 1) % config.checkpoint_every_epochs == 0 or epoch + 1 == config.epochs:
            save_checkpoint(
                checkpoint_path,
                year,
                epoch + 1,
                None,
                0,
                0.0,
                0,
                0,
                model,
                optimizer,
                preprocessor,
                active_ids,
                logger,
                gp_settings,
            )

    model.eval()
    # 冻结模型，用验证年前的近期样本训练树，标签沿用阶段一的时间边界。
    tree_start = bisect_left(panel.dates, date(year - config.lgbols_lookback_years, 1, 1))
    tree_factors, tree_targets = [], []
    factor_started = perf_counter()
    stage(4, 6, f"{year}: Extracting {config.num_factors} factors...", logger)
    with torch.no_grad():
        factor_days = range(max(training_days.start, tree_start), training_days.stop)
        #逐日提取因子
        for day in tqdm(factor_days, desc=f"{year} extracting factors", unit="day",
                        leave=False, disable=None, dynamic_ncols=True, mininterval=1.0):
            ids = panel.pool_rows(day)
            _, x = panel.window(day, config.seq_len, ids)
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
    report(f"{year}: factor extraction completed in {perf_counter() - factor_started:.1f}s.", logger)
    factor_matrix = torch.cat(tree_factors) #拼接交易日数据
    target_vector = torch.cat(tree_targets)

    #开始训练LightGBM
    stage(
        5,
        6,
        f"{year}: Fitting LightGBM GBDT on {len(target_vector):,} samples, "
        f"{factor_matrix.size(1)} factors, {config.lgbols_num_boost_round} rounds...",
        logger,
    )
    lightgbm_started = perf_counter()
    booster = fit_lgbols(factor_matrix, target_vector, config)
    report(f"{year}: LightGBM fitted in {perf_counter() - lightgbm_started:.1f}s.", logger)
    stage(6, 6, f"{year}: Saving prediction scores...", logger)
    prediction_started = perf_counter()
    #生成最终预测
    for day in tqdm(range(stop, end), desc=f"{year} predicting", unit="day",
                    leave=False, disable=None, dynamic_ncols=True, mininterval=1.0):
        with torch.no_grad(): #获取当天股票池
            ids = panel.pool_rows(day)
            if not ids.numel():
                continue
            #提取当天股票弱因子Z，输入LightGBM
            _, x = panel.window(day, config.seq_len, ids)
            x = preprocessor.transform(x).to(device)
            factors = model(x, id_map[ids].to(device))["Z"].cpu()
            scores = torch.from_numpy(
                booster.predict(factors.numpy(), num_threads=config.lgbols_num_threads)
            )
            if not torch.isfinite(scores).all():
                raise RuntimeError(f"Non-finite predictions on {panel.dates[day]}")
        #输出预测结果
        for stock_id, score in zip(ids.tolist(), scores.tolist()):
            yield panel.dates[day].isoformat(), panel.stocks[stock_id], year, score
    report(
        f"{year}: validation prediction completed in {perf_counter() - prediction_started:.1f}s; "
        f"total annual time {perf_counter() - year_started:.1f}s.",
        logger,
    )


# 读取命令行参数，按验证年份汇总预测结果。
def write_parquet_scores(path, rows, arrow, parquet):
    schema = arrow.schema([
        ("date", arrow.timestamp("ns")),
        ("stock_id", arrow.string()),
        ("validation_year", arrow.int64()),
        ("C", arrow.float64()),
    ])
    rows = iter(rows)
    with parquet.ParquetWriter(path, schema) as writer:
        while batch := list(islice(rows, 65536)):
            dates, stock_ids, years, scores = zip(*batch)
            writer.write_table(arrow.Table.from_arrays(
                [arrow.array(dates, type=arrow.string()).cast(schema.field("date").type),
                 arrow.array(stock_ids, type=arrow.string()),
                 arrow.array(years, type=arrow.int64()),
                 arrow.array(scores, type=arrow.float64())],
                schema=schema,
            ))


def main():
    run_started = perf_counter()
    parser = argparse.ArgumentParser(description="Train ASTGNN on rolling day panels.")
    parser.add_argument("--data", type=Path, required=True, help="UTF-8 CSV: date,stock_id,high,open,low,close,vwap,volume")
    parser.add_argument("--mask", type=Path, required=True, help="UTF-8 CSV: date,stock_id,in_pool")
    parser.add_argument("--output", type=Path, required=True, help="New .csv or .parquet file for validation scores")
    defaults = ModelConfig()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(defaults.checkpoint_dir))
    parser.add_argument("--resume", action="store_true", help="Resume annual models from checkpoints.")
    parser.add_argument("--log-file", type=Path, help="Optional UTF-8 run log path.")
    parser.add_argument(
        "--first-prediction-year",
        type=int,
        help="Optional first validation year; defaults to the earliest feasible year.",
    )
    parser.add_argument(
        "--last-prediction-year",
        type=int,
        help="Optional last validation year; defaults to the latest data year.",
    )
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
    output_format = args.output.suffix.lower()
    if output_format not in {".csv", ".parquet"}:
        parser.error("--output must end in .csv or .parquet")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    if output_format == ".parquet":
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            parser.error("Parquet output requires pyarrow.")
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
    require_lightgbm()

    log_path = args.log_file or args.output.with_suffix(".log")
    logger = configure_logger(log_path)
    report(f"Run log: {log_path}", logger)
    stage(1, 6, "Loading day feature panel...", logger)
    panel = DayPanel(args.data, args.mask, logger)
    earliest_feasible_year = panel.dates[0].year + defaults.training_years
    latest_data_year = panel.dates[-1].year
    if earliest_feasible_year > latest_data_year:
        parser.error(
            f"The dataset needs {defaults.training_years} training calendar years "
            "and a validation year."
        )
    first_year = args.first_prediction_year or earliest_feasible_year
    last_year = args.last_prediction_year or latest_data_year
    if first_year < earliest_feasible_year:
        parser.error(
            f"--first-prediction-year must be at least {earliest_feasible_year} for this dataset"
        )
    if last_year > latest_data_year:
        parser.error(f"--last-prediction-year must not exceed {latest_data_year}")
    if first_year > last_year:
        parser.error("--first-prediction-year must not be after --last-prediction-year")
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
    device = resolve_device(config.device, logger)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    def scores():
        for year in range(first_year, last_year + 1):
            checkpoint_path = args.checkpoint_dir / f"astgnn_{year}.pt"
            yield from annual_scores(
                panel, year, config, device, checkpoint_path, args.resume, logger,
                industry_prior_dir=args.industry_prior_dir, lambda_gp=args.lambda_gp,
            )

    if output_format == ".parquet":
        write_parquet_scores(temporary_output, scores(), arrow, parquet)
    else:
        with temporary_output.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("date", "stock_id", "validation_year", "C"))
            writer.writerows(scores())
    os.replace(temporary_output, args.output)
    report(f"Validation scores written to {args.output}", logger)
    report(f"Rolling training completed in {perf_counter() - run_started:.1f}s.", logger)


if __name__ == "__main__":
    main()
