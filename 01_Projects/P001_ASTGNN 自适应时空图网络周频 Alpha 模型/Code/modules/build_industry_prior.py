"""Export monthly industry matrices before running training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modules.industry_relation import IndustryRelations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--industry", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True, help="Daily stock pool Parquet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=int)
    parser.add_argument("--end-date", type=int)
    args = parser.parse_args()

    relations = IndustryRelations(args.industry)
    pool = pd.read_parquet(args.pool, engine="pyarrow")
    if "date" in pool.columns:
        pool = pool.set_index("date", verify_integrity=True)
    pool.index = pd.to_datetime(pool.index.astype(str), format="%Y%m%d")
    if args.start_date is not None:
        pool = pool.loc[pool.index >= pd.to_datetime(str(args.start_date), format="%Y%m%d")]
    if args.end_date is not None:
        pool = pool.loc[pool.index <= pd.to_datetime(str(args.end_date), format="%Y%m%d")]
    if pool.empty or not pool.index.is_unique or not pool.columns.is_unique:
        parser.error("Selected pool must have dates and unique date and stock keys.")
    days = pool.index.sort_values()
    snapshots = days[~days.to_period("M").duplicated()]
    missing_dates = snapshots.difference(relations.classifications.index)
    if len(missing_dates):
        parser.error(f"Industry table is missing {len(missing_dates)} dates, starting {missing_dates[0]}.")

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Later entrants can be sliced from the same snapshot. If their industry
    # was unknown on the snapshot date, the saved pair mask excludes them.
    stock_codes = pool.columns.to_numpy(dtype=str)
    for day in snapshots:
        adjacency, valid_pairs = relations.for_day(day, stock_codes)
        path = args.output_dir / f"{day:%Y%m%d}.npz"
        np.savez_compressed(
            path,
            stock_codes=stock_codes,
            adjacency=np.packbits(adjacency.numpy(), axis=1),
            valid_pairs=np.packbits(valid_pairs.numpy(), axis=1),
        )
    (args.output_dir / "snapshots.json").write_text(json.dumps({
        "start": days[0].date().isoformat(),
        "end": days[-1].date().isoformat(),
        "dates": [day.date().isoformat() for day in snapshots],
    }, indent=2), encoding="utf-8")
    (args.output_dir / "_complete").touch()
    print(f"Saved {len(snapshots)} monthly matrices for {len(days)} trading days to {args.output_dir}")


if __name__ == "__main__":
    main()
