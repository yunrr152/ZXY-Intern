"""Export one dated industry graph in the RSR sector_industry layout."""

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.industry_relation import IndustryRelations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--industry", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    day = pd.Timestamp(args.date)
    classifications = IndustryRelations(args.industry).classifications
    pool = pd.read_parquet(args.pool, engine="pyarrow")
    if "date" in pool.columns:
        pool = pool.set_index("date", verify_integrity=True)
    pool.index = pd.to_datetime(pool.index.astype(str), format="%Y%m%d")
    if not pool.index.is_unique or not pool.columns.is_unique:
        parser.error("The stock pool must have unique dates and stock codes.")
    if day not in pool.index or day not in classifications.index:
        parser.error(f"{args.date} must exist in both the pool and industry tables.")

    membership = pool.loc[day]
    stocks = sorted(membership.index[membership.eq(1).fillna(False)])
    if not stocks:
        parser.error(f"No stocks in the pool on {args.date}.")
    codes = classifications.loc[day].reindex(stocks).dropna().astype("int64")
    groups = {
        str(code): rows.index.tolist()
        for code, rows in codes.groupby(codes, sort=True)
    }
    channels = [code for code, members in groups.items() if len(members) > 1]
    stock_index = {stock: i for i, stock in enumerate(stocks)}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    relations = np.lib.format.open_memmap(
        args.output_dir / "A_SHARE_industry_relation.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(stocks), len(stocks), len(channels) + 1),
    )
    relations[:] = False
    for channel, code in enumerate(channels):
        ids = [stock_index[stock] for stock in groups[code]]
        relations[:, :, channel][np.ix_(ids, ids)] = True
    diagonal = np.arange(len(stocks))
    relations[diagonal, diagonal, -1] = True
    relations.flush()

    (args.output_dir / "A_SHARE_industry_ticker.json").write_text(
        json.dumps(groups, indent=2), encoding="utf-8"
    )
    (args.output_dir / "A_SHARE_industry_axes.json").write_text(
        json.dumps({
            "date": args.date.isoformat(),
            "stock_ids": stocks,
            "channels": [*channels, "self"],
        }, indent=2), encoding="utf-8"
    )
    print(f"Saved {relations.shape} industry relation tensor for {args.date}.")


if __name__ == "__main__":
    main()
