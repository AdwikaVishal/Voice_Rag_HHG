"""Segment 1 — MSMARCO-XI corpus preparation.

Downloads ONE manageable MSMARCO-XI validation Parquet shard, reads it
incrementally with PyArrow, normalizes and filters records, then writes a
small local JSONL corpus plus a statistics file.

Usage (from the backend/ directory):
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --limit 2000 --language hin

Configuration can also come from environment variables:

    MSMARCO_REPO_ID      dataset repo id (default: ai4bharat/MSMARCO-XI)
    MSMARCO_LANGUAGE     language code selecting the {lang}val.parquet shard
    MSMARCO_MAX_RECORDS  number of records to save before stopping
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi

logger = logging.getLogger("prepare_dataset")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RAW_DIR = DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_REPO_ID = "ai4bharat/MSMARCO-XI"
DEFAULT_LANGUAGE = "urd"
DEFAULT_MAX_RECORDS = 5000
DEFAULT_BATCH_SIZE = 2000

OUTPUT_JSONL_NAME = "msmarco_xi_sample.jsonl"
OUTPUT_STATS_NAME = "dataset_stats.json"

RESOLVE_URL = "https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
CHUNK_SIZE = 1 << 20

# Columns needed for the corpus. The nested `meta` group (LLM generation
# parameters) is intentionally not read.
PARQUET_COLUMNS = [
    "source_lang",
    "target_lang",
    "Answer",
    "query_id",
    "query_type",
    "Eng_Query",
    "Eng_Answer",
    "query",
    "passages",
]

_WS_RE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Conservative normalization: collapse whitespace, strip edges.

    Preserves Indic Unicode and the original meaning. No lowercasing.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return _WS_RE.sub(" ", text).strip()


def build_record(row: dict, seq: int) -> dict | None:
    """Turn one raw Parquet row into a cleaned corpus record (or None)."""
    passages_raw = row.get("passages") or {}
    english_list = passages_raw.get("English_passages") or []
    translated_list = passages_raw.get("Translated_passages") or []
    selected_list = passages_raw.get("is_selected") or []

    passages: list[dict] = []
    for idx, english in enumerate(english_list):
        translated = translated_list[idx] if idx < len(translated_list) else None
        selected = selected_list[idx] if idx < len(selected_list) else 0

        english_text = normalize_text(english)
        translated_text = normalize_text(translated)
        if not english_text and not translated_text:
            continue

        passages.append(
            {
                "passage_index": idx,
                "english_text": english_text,
                "translated_text": translated_text,
                "is_selected": int(selected or 0),
            }
        )

    if not passages:
        return None

    query = normalize_text(row.get("query"))
    if not query:
        return None

    return {
        "record_id": f"msmarco_xi_{seq:06d}",
        "query_id": row.get("query_id"),
        "source_lang": row.get("source_lang"),
        "target_lang": row.get("target_lang"),
        "query_type": row.get("query_type"),
        "query": query,
        "english_query": normalize_text(row.get("Eng_Query")),
        "answer": normalize_text(row.get("Answer")),
        "english_answer": normalize_text(row.get("Eng_Answer")),
        "passages": passages,
    }


def download_parquet(
    repo_id: str, source_file: str, dest: Path, expected_size: int | None
) -> str | None:
    """Download the shard with a streaming request; returns sha256 or None."""
    if dest.exists() and dest.stat().st_size > 0:
        if expected_size is None or dest.stat().st_size == expected_size:
            logger.info("Using existing file %s (%d bytes)", dest, dest.stat().st_size)
            return None
        logger.warning("Size mismatch, re-downloading %s", dest)

    url = RESOLVE_URL.format(repo_id=repo_id, path=source_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    logger.info("Downloading %s -> %s", url, dest)
    digest = hashlib.sha256()
    last_logged = -1
    with requests.get(url, stream=True, timeout=(30, 900)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        with tmp.open("wb") as out_file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                out_file.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if total:
                    pct = int(written / total * 100)
                    if pct // 25 > last_logged:
                        last_logged = pct // 25
                        logger.info("Downloaded %d / %d MB (%d%%)", written // (1 << 20), total // (1 << 20), pct)

    tmp.replace(dest)
    logger.info("Download complete: %s (%d bytes)", dest, dest.stat().st_size)
    return digest.hexdigest()


def process_parquet(
    source: Path,
    output_path: Path,
    max_records: int,
    batch_size: int,
    language: str,
    repo_id: str,
    source_file: str,
    sha256: str | None,
) -> dict:
    """Read the shard incrementally and write cleaned JSONL records."""
    parquet_file = pq.ParquetFile(source)
    total_rows = parquet_file.metadata.num_rows
    logger.info(
        "Parquet: %d rows, %d row groups, schema ok",
        total_rows,
        parquet_file.metadata.num_row_groups,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    seq = 1
    processed = 0
    removed = 0
    saved_count = 0
    passage_count = 0
    selected_passage_count = 0
    records_with_selected = 0
    languages: set[str] = set()
    query_ids: set[int] = set()

    with output_path.open("w", encoding="utf-8") as out_file:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=PARQUET_COLUMNS):
            for row in batch.to_pylist():
                processed += 1
                record = build_record(row, seq)
                if record is None:
                    removed += 1
                    continue

                seq += 1
                saved_count += 1
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                passage_count += len(record["passages"])
                record_selected = [p for p in record["passages"] if p["is_selected"] == 1]
                selected_passage_count += len(record_selected)
                if record_selected:
                    records_with_selected += 1
                languages.add(record["source_lang"])
                languages.add(record["target_lang"])
                if record["query_id"] is not None:
                    query_ids.add(record["query_id"])

                if saved_count >= max_records:
                    break
            if saved_count >= max_records:
                break

    return {
        "dataset": repo_id,
        "source_split": Path(source_file).parts[0],
        "source_file": source_file,
        "source_language": language,
        "source_rows_total": total_rows,
        "source_sha256": sha256,
        "max_records_target": max_records,
        "records_processed": processed,
        "records_saved": saved_count,
        "records_removed": removed,
        "passage_count": passage_count,
        "selected_passage_count": selected_passage_count,
        "records_with_selected": records_with_selected,
        "unique_query_ids": len(query_ids),
        "languages_observed": sorted(l for l in languages if l),
        "output_file": display_path(output_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def display_path(path: Path) -> str:
    """Path relative to the backend dir when possible, else absolute."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MSMARCO-XI local corpus")
    parser.add_argument("--repo-id", default=os.environ.get("MSMARCO_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument(
        "--language",
        default=os.environ.get("MSMARCO_LANGUAGE", DEFAULT_LANGUAGE),
        help="Language code selecting the {lang}val.parquet shard (default: %(default)s)",
    )
    parser.add_argument("--source", default=None, help="Explicit shard path inside the repo")
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("MSMARCO_MAX_RECORDS", DEFAULT_MAX_RECORDS)),
        help="Maximum number of records to save (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.source:
        source_file = args.source
    else:
        source_file = f"validation/{args.language}val.parquet"

    api = HfApi()
    repo_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset"))
    if source_file not in repo_files:
        logger.error(
            "%s not found in %s. Available validation files:\n  %s",
            source_file,
            args.repo_id,
            "\n  ".join(sorted(f for f in repo_files if f.startswith("validation"))),
        )
        sys.exit(1)

    expected_size = None
    for info in api.get_paths_info(
        repo_id=args.repo_id, paths=[source_file], repo_type="dataset"
    ):
        expected_size = info.size
    logger.info(
        "Source %s (%s) — %s",
        source_file,
        args.repo_id,
        f"{expected_size / 1e6:.1f} MB" if expected_size else "unknown size",
    )

    raw_path = args.raw_dir / source_file
    sha256 = None
    if args.force_download or not raw_path.exists():
        sha256 = download_parquet(args.repo_id, source_file, raw_path, expected_size)
    else:
        logger.info("Using existing file %s", raw_path)

    output_jsonl = args.processed_dir / OUTPUT_JSONL_NAME
    output_stats = args.processed_dir / OUTPUT_STATS_NAME

    stats = process_parquet(
        source=raw_path,
        output_path=output_jsonl,
        max_records=args.limit,
        batch_size=args.batch_size,
        language=args.language,
        repo_id=args.repo_id,
        source_file=source_file,
        sha256=sha256,
    )

    output_stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    logger.info("Wrote %s", display_path(output_jsonl))
    logger.info("Wrote %s", display_path(output_stats))
    logger.info(
        "records_processed=%d records_saved=%d records_removed=%d passage_count=%d selected_passage_count=%d",
        stats["records_processed"],
        stats["records_saved"],
        stats["records_removed"],
        stats["passage_count"],
        stats["selected_passage_count"],
    )


if __name__ == "__main__":
    main()
