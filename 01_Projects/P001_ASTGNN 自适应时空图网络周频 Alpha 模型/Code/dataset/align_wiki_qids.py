"""Align a Wikidata Query Service CSV with the A-share QID review template."""

import argparse
from pathlib import Path

import pandas as pd


EXCHANGES = {"Q517750": "XSHE", "Q739514": "XSHG"}
ENTITY_ID = r"(?:https?://(?:www\.)?wikidata\.org/entity/)?(Q[1-9]\d*)"


def align(raw_path, template_path, output_dir):
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    template = pd.read_csv(template_path, dtype=str, keep_default_na=False)
    source_columns = ["company", "ticker", "exchange"]
    raw = raw[source_columns].apply(lambda column: column.str.strip()).drop_duplicates()

    raw["qid"] = raw["company"].str.extract(f"^{ENTITY_ID}$", expand=False)
    suffix = raw["exchange"].str.extract(f"^{ENTITY_ID}$", expand=False).map(EXCHANGES)
    raw["stock_id"] = raw["ticker"] + "." + suffix
    valid = raw["ticker"].str.fullmatch(r"[0-9]{6}") & raw["qid"].notna() & suffix.notna()
    candidates = raw.loc[valid, ["stock_id", "qid"]].drop_duplicates()
    counts = candidates.groupby("stock_id")["qid"].nunique()
    template_ids = template["stock_id"].str.strip()
    duplicate_ids = template_ids[template_ids.duplicated(keep=False)]

    raw["issue"] = ""
    raw.loc[~valid, "issue"] = "invalid_ticker_exchange_or_qid"
    raw.loc[valid & ~raw["stock_id"].isin(template_ids), "issue"] = "stock_not_in_template"
    matched = valid & raw["stock_id"].isin(template_ids)
    raw.loc[matched & raw["stock_id"].map(counts).gt(1), "issue"] = "multiple_qids"
    raw.loc[matched & raw["stock_id"].isin(duplicate_ids), "issue"] = "multiple_template_rows"

    unique = candidates[
        candidates["stock_id"].map(counts).eq(1)
        & ~candidates["stock_id"].isin(duplicate_ids)
    ].set_index("stock_id")["qid"]
    proposed = template_ids.map(unique)
    preserve = template["qid"].str.strip().ne("")
    mismatch = preserve & proposed.notna() & template["qid"].str.strip().ne(proposed)
    raw.loc[raw["stock_id"].isin(template_ids[mismatch]), "issue"] = "existing_qid_differs"

    fill = (
        proposed.notna() & ~preserve
        & template["match_status"].str.strip().str.lower().eq("pending")
    )
    template.loc[fill, "qid"] = proposed[fill]
    add_source = fill & template["evidence"].str.strip().eq("")
    template.loc[add_source, "evidence"] = "https://www.wikidata.org/wiki/" + proposed[add_source]

    output_dir.mkdir(parents=True, exist_ok=False)
    template.to_csv(output_dir / "stock_qids.csv", index=False, encoding="utf-8-sig")
    raw.loc[raw["issue"].ne("")].to_csv(
        output_dir / "alignment_issues.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Prefilled {fill.sum()} pending QIDs. Kept existing review statuses.")
    for issue, size in raw.loc[raw["issue"].ne(""), "issue"].value_counts().items():
        print(f"{issue}: {size} raw rows")
    print(f"Results: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Wikidata CSV export.")
    parser.add_argument("--template", type=Path, required=True, help="Local stock_qids.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory.")
    args = parser.parse_args()
    align(args.raw, args.template, args.output_dir)


if __name__ == "__main__":
    main()
