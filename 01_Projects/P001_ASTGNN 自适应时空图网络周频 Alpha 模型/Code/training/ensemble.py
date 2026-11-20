import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from config import ModelConfig
from data.build_day_dataset import build_dataset
from data.day import DayPanel
from training.lgbols import require_lightgbm
from training.train import annual_scores, configure_logger, resolve_device


PREDICTION_KEYS = ("date", "stock_id", "validation_year")


def mean_predictions(predictions):
    scores = [
        frame.set_index(list(PREDICTION_KEYS), verify_integrity=True)["C"].sort_index()
        for frame in predictions
    ]
    if any(not score.index.equals(scores[0].index) for score in scores[1:]):
        raise ValueError("Prediction keys differ across seeds; cannot average scores.")
    return (sum(scores) / len(scores)).rename("C").reset_index()


def prepare_dataset(args, config):
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=args.resume)
    settings = {
        "source_root": str(args.source_root.resolve()),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "first_prediction_year": args.first_prediction_year,
        "last_prediction_year": args.last_prediction_year,
        "config": asdict(config),
    }
    settings = json.loads(json.dumps(settings))
    manifest = directory / "ensemble_run.json"
    features = directory / "day_features.csv"
    mask = directory / "day_pool_mask.csv"
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if recorded["settings"] != settings:
            raise ValueError("Resume settings differ from the original ensemble run.")
        if any(not path.is_file() or path.stat().st_size != recorded["input_sizes"][path.name]
               for path in (features, mask)):
            raise ValueError("Saved ensemble input CSVs are missing or have changed size.")
        return features, mask, True

    legacy = (directory / "ensemble.log").is_file()
    if legacy:
        if not args.resume or not all(path.is_file() for path in (features, mask)):
            raise ValueError("Legacy ensemble resume requires both original input CSVs.")
        print("Resuming a legacy run: verify that the original arguments and config are unchanged.")
    else:
        if (directory / "checkpoints").exists() or any(directory.glob("predict*.parquet")):
            raise ValueError("Cannot establish which data produced the existing ensemble results.")
        with TemporaryDirectory(prefix="dataset-", dir=directory) as temporary:
            args.features_output = Path(temporary) / features.name
            args.mask_output = Path(temporary) / mask.name
            build_dataset(args)
            args.features_output.replace(features)
            args.mask_output.replace(mask)

    temporary_manifest = manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps({
        "settings": settings,
        "input_sizes": {path.name: path.stat().st_size for path in (features, mask)},
    }, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest)
    return features, mask, False


def write_parquet(frame, path):
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(path)


def main():
    defaults = ModelConfig()
    parser = argparse.ArgumentParser(description="Build daily data and train a three-seed ensemble.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--start-date", type=int, required=True, help="Inclusive YYYYMMDD.")
    parser.add_argument("--end-date", type=int, required=True, help="Inclusive YYYYMMDD.")
    parser.add_argument("--first-prediction-year", type=int, required=True)
    parser.add_argument("--last-prediction-year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume using the original run directory.")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    args = parser.parse_args()
    if args.epochs < 1 or args.first_prediction_year > args.last_prediction_year:
        parser.error("Epochs must be positive and the prediction-year range must be ordered.")

    import pyarrow

    require_lightgbm()
    device = resolve_device(args.device)
    run_config = replace(defaults, use_resnet=False, epochs=args.epochs, device=args.device)
    features, mask, reuse_predictions = prepare_dataset(args, run_config)
    logger = configure_logger(args.output_dir / "ensemble.log")
    panel = DayPanel(features, mask, logger)
    if (
        args.first_prediction_year < panel.dates[0].year + defaults.training_years
        or args.last_prediction_year > panel.dates[-1].year
    ):
        parser.error("Prediction years must be covered by the panel, with ten prior training years.")

    predictions = []
    for index, seed in enumerate(defaults.ensemble_seeds, start=1):
        print(f"Training ensemble member {index}/{len(defaults.ensemble_seeds)} with seed {seed}.")
        config = replace(run_config, random_seed=seed)
        checkpoint_dir = args.output_dir / "checkpoints" / f"seed_{seed}"
        parquet_path = args.output_dir / f"predict{index}.parquet"
        completion_path = parquet_path.with_suffix(".done")
        if reuse_predictions and completion_path.is_file() and parquet_path.is_file():
            print(f"Reusing completed predictions for seed {seed}.")
            predictions.append(pd.read_parquet(parquet_path, engine="pyarrow"))
            continue
        csv_path = args.output_dir / f"predict{index}.csv.tmp"
        with csv_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow((*PREDICTION_KEYS, "C"))
            for year in range(args.first_prediction_year, args.last_prediction_year + 1):
                writer.writerows(annual_scores(
                    panel, year, config, device, checkpoint_dir / f"astgnn_{year}.pt",
                    resume=args.resume, logger=logger,
                ))
        frame = pd.read_csv(csv_path, dtype={"stock_id": "string"}, parse_dates=["date"])
        if frame.empty:
            raise ValueError(f"Seed {seed} produced no predictions.")
        write_parquet(frame, parquet_path)
        completion_path.touch()
        csv_path.unlink()
        predictions.append(frame)

    write_parquet(mean_predictions(predictions), args.output_dir / "predict.parquet")


if __name__ == "__main__":
    main()
