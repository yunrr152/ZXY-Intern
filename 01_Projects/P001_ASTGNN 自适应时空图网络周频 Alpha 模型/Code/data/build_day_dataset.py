"""Build raw day-model inputs and an aligned daily pool mask.

Example (review before running):
    conda run -n astgnn python -B -m data.build_day_dataset \
        --source-root data/data \
        --start-date 20070104 --end-date 20261117 \
        --features-output data/output/day_features.csv \
        --mask-output data/output/day_pool_mask.csv

The feature file contains every stock-day whose tradeStatus equals 1.  The
mask file has the same (date, stock_id) keys and indicates membership of
advanceWholeMarketNew on that date.  No winsorisation, standardisation,
missing-value filling, or target calculation is performed here.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm


RAW_FEATURES = {
    "high": "highA",
    "open": "openA",
    "low": "lowA",
    "close": "closeA",
    "vwap": "vwapA",
    "volume": "volumeA",
}
REQUIRED_FIELDS = ("di", "tradeStatus", *RAW_FEATURES.values())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build raw day features and an advanceWholeMarketNew mask."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing LX_STK_BASIC_DAILY and LX_STK_POOL_BASIC.",
    )
    parser.add_argument(
        "--start-date", type=int, required=True, help="Inclusive YYYYMMDD."
    )
    parser.add_argument(
        "--end-date", type=int, required=True, help="Inclusive YYYYMMDD."
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        required=True,
        help="New CSV: date,stock_id,high,open,low,close,vwap,volume",
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        required=True,
        help="New CSV: date,stock_id,in_pool",
    )
    return parser.parse_args()


def iso_date(trade_date):
    """Convert an integer YYYYMMDD trade date to the DayPanel CSV format."""
    value = f"{int(trade_date):08d}"
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def read_field(folder, field):
    path = folder / f"{field}.dat"
    if not path.is_file():
        raise FileNotFoundError(f"Missing required field: {path}")
    dtype = "<i4" if field == "di" else "<f4"
    return np.fromfile(path, dtype=dtype)


def read_stock(folder):
    """Read only fields needed by the current six-raw-feature definition."""
    values = {field: read_field(folder, field) for field in REQUIRED_FIELDS}
    length = len(values["di"])
    mismatched = [field for field, column in values.items() if len(column) != length]
    if mismatched:
        raise ValueError(
            f"{folder.name} has fields with lengths different from di.dat: {mismatched}"
        )
    return values


def validate_arguments(args):
    if args.start_date > args.end_date:
        raise ValueError("--start-date must not be after --end-date")
    if args.features_output.resolve() == args.mask_output.resolve():
        raise ValueError("Feature and mask outputs must be different files")
    for output in (args.features_output, args.mask_output):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    daily_root = args.source_root / "LX_STK_BASIC_DAILY"
    pool_path = args.source_root / "LX_STK_POOL_BASIC" / "advanceWholeMarketNew.parquet"
    if not daily_root.is_dir():
        raise FileNotFoundError(f"Daily-data directory not found: {daily_root}")
    if not pool_path.is_file():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")
    return daily_root, pool_path


def build_dataset(args):
    build_started = perf_counter()
    daily_root, pool_path = validate_arguments(args)
    # The source parquet is the authoritative time-varying pool membership mask.
    print("Loading advanceWholeMarketNew pool mask...")
    pool_started = perf_counter()
    pool = pd.read_parquet(pool_path)
    pool.index = pd.Index(pool.index.astype("int64"), name="trade_date")
    pool = pool.loc[(pool.index >= args.start_date) & (pool.index <= args.end_date)]
    print(f"Pool mask loaded in {perf_counter() - pool_started:.1f}s.")

    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    stock_folders = sorted(folder for folder in daily_root.iterdir() if folder.is_dir())
    print(f"Reading {len(stock_folders):,} stock histories and writing CSV outputs...")
    written_rows = 0
    pooled_rows = 0

    with (
        args.features_output.open("x", newline="", encoding="utf-8") as feature_file,
        args.mask_output.open("x", newline="", encoding="utf-8") as mask_file,
    ):
        feature_writer = csv.writer(feature_file)
        mask_writer = csv.writer(mask_file)
        feature_writer.writerow(("date", "stock_id", *RAW_FEATURES))
        mask_writer.writerow(("date", "stock_id", "in_pool"))

        with tqdm(stock_folders, desc="Reading stock histories", unit="stock") as progress:
            for folder in progress:
                values = read_stock(folder)
                selected = (
                    (values["di"] >= args.start_date)
                    & (values["di"] <= args.end_date)
                    & (values["tradeStatus"] == 1)
                )
                if selected.any():
                    dates = values["di"][selected]
                    # Stocks absent from the pool table, dates outside it, and non-1 values are out of pool.
                    if folder.name in pool.columns:
                        membership = pool[folder.name].reindex(dates, fill_value=0).eq(1).to_numpy()
                    else:
                        membership = np.zeros(len(dates), dtype=bool)

                    raw_columns = [values[source][selected] for source in RAW_FEATURES.values()]
                    for row_index, trade_date in enumerate(dates):
                        date_text = iso_date(trade_date)
                        feature_writer.writerow(
                            (date_text, folder.name, *(column[row_index] for column in raw_columns))
                        )
                        mask_writer.writerow((date_text, folder.name, int(membership[row_index])))

                    written_rows += len(dates)
                    pooled_rows += int(membership.sum())
                progress.set_postfix(rows=f"{written_rows:,}", in_pool=f"{pooled_rows:,}")

    print(f"Wrote {written_rows:,} normal-trading feature rows to {args.features_output}")
    print(f"Wrote {written_rows:,} aligned mask rows to {args.mask_output}")
    print(f"advanceWholeMarketNew member rows: {pooled_rows:,}")
    print(f"Dataset build completed in {perf_counter() - build_started:.1f}s.")


def main():
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
