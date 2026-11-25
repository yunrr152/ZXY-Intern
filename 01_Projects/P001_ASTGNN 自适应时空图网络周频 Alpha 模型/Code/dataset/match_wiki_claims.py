"""Match both QIDs in a Wikidata relation export to A-share stock codes."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re

from dataset.align_wiki_qids import ENTITY_ID


REVIEW_DIR = Path(__file__).resolve().parent / "output" / "wiki_review"
FIELDS = (
    "source_stock", "source_qid", "source_match_status", "property_id",
    "target_stock", "target_qid", "target_match_status", "start", "end", "reference",
)


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def match_claims(claims_path, mapping_path, output_path):
    mapping = defaultdict(dict)
    for row in read_csv(mapping_path, ("stock_id", "qid", "match_status")):
        qid = row["qid"].strip()
        stock = row["stock_id"].strip()
        if qid and stock:
            status = row["match_status"].strip()
            if mapping[qid].get(stock) != "confirmed":
                mapping[qid][stock] = status

    claims = read_csv(claims_path, ("source", "property", "target"))
    results = {}
    pairs = set()
    unmatched = 0
    for row in claims:
        source = re.fullmatch(ENTITY_ID, row["source"].strip())
        target = re.fullmatch(ENTITY_ID, row["target"].strip())
        if not source or not target or source[1] not in mapping or target[1] not in mapping:
            unmatched += 1
            continue
        if source[1] == target[1]:
            continue
        for left, left_status in mapping[source[1]].items():
            for right, right_status in mapping[target[1]].items():
                if left == right:
                    continue
                record = (
                    left, source[1], left_status, row["property"].strip(),
                    right, target[1], right_status, row.get("start", ""),
                    row.get("end", ""), row.get("reference", ""),
                )
                results[record] = None
                pairs.add(tuple(sorted((left, right))))

    with output_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(results)
    for record in results:
        left, left_qid, left_status, prop, right, right_qid, right_status, *_ = record
        print(f"{left} ({left_qid}, {left_status}) --{prop}--> "
              f"{right} ({right_qid}, {right_status})")
    if not results:
        print("No relations matched two different A-share stocks.")
    print(f"{len(claims)} input rows, {unmatched} with an unmapped endpoint.")
    print(f"{len(results)} matched rows, {len(pairs)} distinct undirected stock pairs.")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claims", type=Path, help="CSV path or filename under wiki_review.")
    parser.add_argument("--mapping", type=Path, default=REVIEW_DIR / "aligned" / "stock_qids.csv")
    parser.add_argument("--output", type=Path, help="New CSV. Defaults to <input>_matched.csv.")
    args = parser.parse_args()
    claims = args.claims
    if not claims.is_file() and claims.parent == Path("."):
        claims = REVIEW_DIR / claims
    output = args.output or claims.with_name(f"{claims.stem}_matched.csv")
    try:
        match_claims(claims, args.mapping, output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
