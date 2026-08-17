"""Segment 1 — validate the processed MSMARCO-XI JSONL corpus.

Usage (from the backend/ directory):
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --path data/processed/msmarco_xi_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("validate_dataset")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PATH = BASE_DIR / "data" / "processed" / "msmarco_xi_sample.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the processed MSMARCO-XI corpus")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--show", type=int, default=3, help="First N records to print")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    path = args.path
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    record_ids: set[str] = set()
    total = 0
    selected = 0
    empty_query = 0
    empty_passages = 0
    non_ascii_ok = 0

    with path.open(encoding="utf-8") as in_file:
        for line in in_file:
            record = json.loads(line)
            total += 1

            if record["record_id"] in record_ids:
                logger.error("Duplicate record_id: %s", record["record_id"])
                sys.exit(1)
            record_ids.add(record["record_id"])

            if not record["query"].strip():
                empty_query += 1
            if not record["passages"]:
                empty_passages += 1
            selected += sum(1 for p in record["passages"] if p["is_selected"] == 1)

            if not record["query"].isascii():
                non_ascii_ok += 1

            if total == 1:
                first = record

    logger.info("Total records: %d", total)
    logger.info("Unique record_id: %d", len(record_ids))
    logger.info("Records with empty query: %d", empty_query)
    logger.info("Records with no passages: %d", empty_passages)
    logger.info("Selected passages: %d", selected)
    logger.info("Records with non-ASCII (Indic) query text: %d", non_ascii_ok)

    assert total == len(record_ids), "record_id uniqueness check failed"
    assert empty_query == 0, "empty queries found"
    assert empty_passages == 0, "records without passages found"

    logger.info("All checks passed for %s", path)

    if args.show:
        print("\nFirst %d record(s):" % args.show)
        with path.open(encoding="utf-8") as in_file:
            for _ in range(args.show):
                print(json.dumps(json.loads(next(in_file)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
