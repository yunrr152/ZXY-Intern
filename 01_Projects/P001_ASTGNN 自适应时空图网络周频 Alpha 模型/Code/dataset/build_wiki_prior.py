"""Prepare A-share Wikidata review tables and export monthly graph priors."""

import argparse
from collections import defaultdict
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROPERTIES = {"P127", "P355", "P749"}
PILOT_PATHS = {"P127", "P749"}
CLAIM_COLUMNS = ["source_qid", "property_id", "target_qid",
                 "valid_from", "valid_to", "known_from", "evidence"]
NEGATIVE_COLUMNS = ["source_stock", "target_stock",
                    "valid_from", "valid_to", "known_from", "evidence"]


def prepare(args):
    fields = ["code", "partyID", "secFullName", "secShortName",
              "ticker", "exchangeCD", "listDate", "delistDate"]
    master = pd.read_parquet(args.master, columns=fields).rename(columns={"code": "stock_id"})
    master = master[master["stock_id"].str.endswith((".XSHE", ".XSHG"), na=False)].copy()
    master["qid"] = ""
    master["match_status"] = "pending"
    master["evidence"] = ""
    args.output_dir.mkdir(parents=True, exist_ok=False)
    master.to_csv(args.output_dir / "stock_qids.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=CLAIM_COLUMNS).to_csv(args.output_dir / "claims.csv", index=False)
    pd.DataFrame(columns=NEGATIVE_COLUMNS).to_csv(args.output_dir / "negative_pairs.csv", index=False)
    print(f"Prepared {len(master)} issuer records in {args.output_dir}")


def read_evidence(path, columns, calendar_end):
    table = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    table = table.apply(lambda col: col.str.strip())
    required = [col for col in columns if col != "valid_to"]
    if table[required].eq("").any().any():
        raise ValueError(f"{path}: all fields except valid_to are required.")
    start = pd.to_datetime(table["valid_from"], format="%Y-%m-%d")
    known = pd.to_datetime(table["known_from"], format="%Y-%m-%d")
    end = pd.to_datetime(table["valid_to"].replace("", pd.NA), format="%Y-%m-%d")
    if (end.notna() & (end <= start)).any():
        raise ValueError(f"{path}: valid_to must be later than valid_from.")
    table["first_day"] = start.where(start >= known, known)
    table["end_day"] = end.fillna(calendar_end + pd.Timedelta(days=1))
    return table


def wiki_edges(claims, qid_to_rows, second_order):
    pairs = set()
    groups = defaultdict(set)
    for claim in claims.itertuples(index=False):
        sources = qid_to_rows.get(claim.source_qid, ())
        targets = qid_to_rows.get(claim.target_qid, ())
        if claim.source_qid != claim.target_qid:
            pairs.update(tuple(sorted(pair)) for pair in product(sources, targets))
        if second_order and claim.property_id in PILOT_PATHS and sources:
            groups[claim.target_qid].add(claim.source_qid)
    for issuers in groups.values():
        for left, right in combinations(issuers, 2):
            pairs.update(tuple(sorted(pair)) for pair in product(qid_to_rows[left], qid_to_rows[right]))
    return pairs


def pack_pairs(pairs, size):
    packed = np.zeros((size, (size + 7) // 8), dtype=np.uint8)
    if pairs:
        left, right = np.array(sorted(pairs), dtype=np.intp).T
        rows = np.concatenate((left, right))
        columns = np.concatenate((right, left))
        bits = (1 << (7 - columns % 8)).astype(np.uint8)
        np.bitwise_or.at(packed, (rows, columns // 8), bits)
    return packed


def build(args):
    pool = pd.read_parquet(args.pool)
    if "date" in pool.columns:
        pool = pool.set_index("date", verify_integrity=True)
    days = pd.to_datetime(pool.index.astype(str), format="%Y%m%d").sort_values()
    if args.start_date:
        days = days[days >= pd.Timestamp(args.start_date)]
    if args.end_date:
        days = days[days <= pd.Timestamp(args.end_date)]
    if days.empty or not days.is_unique or not pool.columns.is_unique:
        raise ValueError("Select a nonempty calendar and a pool with unique stock codes and dates.")
    stocks = pool.columns.to_numpy(dtype=str)
    stock_index = {stock: index for index, stock in enumerate(stocks)}
    del pool

    mapping = pd.read_csv(args.mapping, dtype=str, keep_default_na=False)
    mapping = mapping.loc[mapping["match_status"].str.strip().str.lower().eq("confirmed")].copy()
    for col in ("stock_id", "qid", "evidence"):
        mapping[col] = mapping[col].str.strip()
    mapping = mapping[mapping["stock_id"].isin(stock_index)]
    if (mapping.empty or not mapping["qid"].str.fullmatch(r"Q[1-9]\d*").all()
            or mapping["evidence"].eq("").any()):
        raise ValueError("Supply confirmed pool-stock mappings with valid QIDs and evidence.")
    mapping = mapping[["stock_id", "qid"]].drop_duplicates()
    if mapping["stock_id"].duplicated().any():
        raise ValueError("A stock code has multiple confirmed issuer QIDs.")
    qid_to_rows = defaultdict(list)
    for row in mapping.itertuples(index=False):
        qid_to_rows[row.qid].append(stock_index[row.stock_id])

    claims = read_evidence(args.claims, CLAIM_COLUMNS, days[-1])
    if not claims["property_id"].isin(PROPERTIES).all():
        raise ValueError("Supported claim properties are P127, P355 and P749.")
    for col in ("source_qid", "target_qid"):
        if not claims[col].str.fullmatch(r"Q[1-9]\d*").all():
            raise ValueError(f"Malformed QID in {col}.")
    negatives = None
    if args.negative_pairs:
        negatives = read_evidence(args.negative_pairs, NEGATIVE_COLUMNS, days[-1])
        for col in ("source_stock", "target_stock"):
            if not negatives[col].isin(mapping["stock_id"]).all():
                raise ValueError(f"{col} must contain confirmed pool-stock mappings.")
        if negatives["source_stock"].eq(negatives["target_stock"]).any():
            raise ValueError("A stock cannot be a negative pair with itself.")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = days[~days.to_period("M").duplicated()]
    counts = []
    for day in snapshots:
        active = claims[(claims.first_day <= day) & (day < claims.end_day)]
        positive = wiki_edges(active, qid_to_rows, args.second_order)
        negative = set()
        if negatives is not None:
            reviewed = negatives[(negatives.first_day <= day) & (day < negatives.end_day)]
            negative = {
                tuple(sorted((stock_index[row.source_stock], stock_index[row.target_stock])))
                for row in reviewed.itertuples(index=False)
            }
        if positive & negative:
            raise ValueError(f"Conflicting positive and negative labels on {day.date()}.")
        adjacency = pack_pairs(positive, len(stocks))
        valid_pairs = adjacency | pack_pairs(negative, len(stocks))
        np.savez_compressed(args.output_dir / f"{day:%Y%m%d}.npz",
                            stock_codes=stocks, adjacency=adjacency, valid_pairs=valid_pairs)
        counts.append({"date": day.date().isoformat(), "positive_pairs": len(positive),
                       "negative_pairs": len(negative),
                       "connected_stocks": len({node for pair in positive for node in pair})})
        if args.second_order:
            direct = wiki_edges(active, qid_to_rows, False)
            counts[-1]["second_order_only_pairs"] = len(positive - direct)
    (args.output_dir / "snapshots.json").write_text(json.dumps({
        "start": days[0].date().isoformat(),
        "end": days[-1].date().isoformat(),
        "dates": [entry["date"] for entry in counts],
        "source": "wikidata",
        "confirmed_stock_mappings": len(mapping),
        "second_order": args.second_order,
        "coverage": counts,
    }, indent=2), encoding="utf-8")
    (args.output_dir / "_complete").touch()
    print(f"Saved {len(snapshots)} snapshots from {len(mapping)} confirmed stock mappings.")
    if args.second_order:
        second_counts = [row["second_order_only_pairs"] for row in counts]
        print(f"New second-order edges in {sum(count > 0 for count in second_counts)} snapshots "
              f"(maximum {max(second_counts)} per snapshot).")
    if not any(row["positive_pairs"] for row in counts):
        print("No supported historical edges in this export.")
    if not any(row["negative_pairs"] for row in counts):
        print("The export has no reviewed negatives. Its loss would supervise positive edges only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Create CSVs for manual review.")
    prep.add_argument("--master", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.set_defaults(run=prepare)
    export = commands.add_parser("build", help="Export priors from reviewed CSVs.")
    for option in ("mapping", "claims", "pool", "output-dir"):
        export.add_argument(f"--{option}", type=Path, required=True)
    export.add_argument("--negative-pairs", type=Path)
    export.add_argument("--second-order", action="store_true",
                        help="Include validated shared-owner or shared-parent paths.")
    export.add_argument("--start-date", help="Inclusive YYYY-MM-DD.")
    export.add_argument("--end-date", help="Inclusive YYYY-MM-DD.")
    export.set_defaults(run=build)
    output_root = Path(__file__).resolve().parent / "output"
    review = output_root / "wiki_review"
    preset = commands.add_parser("build-2017", help="Export the 2017 training prior with second-order edges.")
    preset.add_argument("--output-dir", type=Path,
                        default=output_root / "wiki_prior_direct_plus_second_2007_2016")
    preset.set_defaults(
        run=build,
        mapping=review / "aligned" / "stock_qids.csv",
        claims=review / "claims.csv",
        pool=output_root.parent / "data" / "LX_STK_POOL_BASIC" / "advanceWholeMarketNew.parquet",
        negative_pairs=None,
        start_date="2007-01-04",
        end_date="2016-12-31",
        second_order=True,
    )
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
