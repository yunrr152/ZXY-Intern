"""Create a review queue of listed companies sharing a Wikidata parent or owner."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from dataset.align_wiki_qids import ENTITY_ID
from dataset.build_wiki_prior import CLAIM_COLUMNS, PILOT_PATHS
from dataset.match_wiki_claims import REVIEW_DIR, read_csv


WIKI_ITEM = "https://www.wikidata.org/wiki/"
FIELDS = (
    "priority", "source_count", "parent_claim_count", "target_qid", "target_url",
    "source_stock_ids", "source_names", "source_qid", "source_url",
    "source_match_status", "property_id", "wikidata_records",
    "review_status", "valid_from", "valid_to", "known_from", "evidence", "review_note",
)


def priority(parent_count):
    if parent_count >= 2:
        return "01_shared_parent_candidate"
    if parent_count == 1:
        return "02_parent_and_owner_candidate"
    return "03_shared_owner_candidate"


def prepare_review(raw_path, mapping_path, claims_path, output_path):
    mapping = defaultdict(list)
    for row in read_csv(mapping_path, ("stock_id", "secShortName", "qid", "match_status")):
        qid = row["qid"].strip()
        if re.fullmatch(ENTITY_ID, qid) and row["match_status"].strip().lower() != "rejected":
            mapping[qid].append(row)

    existing = defaultdict(list)
    for row in read_csv(claims_path, CLAIM_COLUMNS):
        key = tuple(row[col].strip() for col in CLAIM_COLUMNS[:3])
        existing[key].append(row)

    groups = defaultdict(lambda: defaultdict(set))
    for row in read_csv(raw_path, ("source", "property", "target")):
        source = re.fullmatch(ENTITY_ID, (row["source"] or "").strip())
        target = re.fullmatch(ENTITY_ID, (row["target"] or "").strip())
        prop = (row["property"] or "").strip()
        if (not source or not target or source[1] == target[1]
                or source[1] not in mapping or prop not in PILOT_PATHS):
            continue
        evidence = tuple((row.get(col) or "").strip() for col in ("start", "end", "reference"))
        groups[target[1]][source[1], prop].add(evidence)

    candidates = []
    for target, statements in groups.items():
        sources = {qid for qid, _ in statements}
        if len(sources) < 2:
            continue
        parents = {qid for qid, prop in statements if prop == "P749"}
        candidates.append((priority(len(parents)), target, len(sources), len(parents), statements))

    written = 0
    with output_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for rank, target, count, parent_count, statements in sorted(candidates):
            for (source, prop), variants in sorted(statements.items()):
                matches = mapping[source]
                approved = existing.get((source, prop, target), [])
                dates = approved[0] if len(approved) == 1 else {}
                records = [
                    dict(zip(("start", "end", "reference"), variant))
                    for variant in sorted(variants)
                ]
                writer.writerow({
                    "priority": rank,
                    "source_count": count,
                    "parent_claim_count": parent_count,
                    "target_qid": target,
                    "target_url": WIKI_ITEM + target,
                    "source_stock_ids": " | ".join(sorted({row["stock_id"] for row in matches})),
                    "source_names": " | ".join(sorted({row["secShortName"] for row in matches})),
                    "source_qid": source,
                    "source_url": WIKI_ITEM + source,
                    "source_match_status": " | ".join(sorted({row["match_status"] for row in matches})),
                    "property_id": prop,
                    "wikidata_records": json.dumps(records, ensure_ascii=False),
                    "review_status": "already_in_claims" if approved else "pending",
                    "valid_from": dates.get("valid_from", ""),
                    "valid_to": dates.get("valid_to", ""),
                    "known_from": dates.get("known_from", ""),
                    "evidence": dates.get("evidence", ""),
                    "review_note": "",
                })
                written += 1

    parent_groups = sum(rank == "01_shared_parent_candidate" for rank, *_ in candidates)
    print(f"{len(candidates)} shared targets, {parent_groups} with at least two P749 sources.")
    print(f"{written} distinct statements. Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="Wikidata relations CSV path or filename under wiki_review.")
    parser.add_argument("--mapping", type=Path, default=REVIEW_DIR / "aligned" / "stock_qids.csv")
    parser.add_argument("--claims", type=Path, default=REVIEW_DIR / "claims.csv")
    parser.add_argument("--output", type=Path, help="New CSV. Defaults to <input>_second_order_review.csv.")
    args = parser.parse_args()
    raw = args.raw
    if not raw.is_file() and raw.parent == Path("."):
        raw = REVIEW_DIR / raw
    output = args.output or raw.with_name(f"{raw.stem}_second_order_review.csv")
    try:
        prepare_review(raw, args.mapping, args.claims, output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

