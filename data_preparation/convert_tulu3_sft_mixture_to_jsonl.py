#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Convert the Tulu 3 SFT mixture parquet shards to prompt-completion JSONL.

The source rows contain ``id``, ``messages`` and ``source``. The output keeps
only the two text fields expected by Megatron-Bridge's local-jsonl
prompt-completion preset:

    {"input": "<prompt>", "output": "<completion>"}

The final assistant message is used as the completion. Earlier messages are
kept as prompt context for multi-turn conversations. A single user/assistant
conversation therefore has exactly the same shape as the example above.
The converter reads one row group at a time, so it does not load the complete
dataset into memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise SystemExit("This program requires pyarrow. Install it with: pip install pyarrow") from exc


REQUIRED_COLUMNS = {"id", "messages", "source"}
OUTPUT_FILENAME = "training.jsonl"


def format_prompt(messages: list[dict[str, Any]]) -> str:
    """Convert prompt-side chat messages into one text prompt.

    Keep the common single-user case unmodified. For multi-turn rows, role
    labels retain enough structure for the model to distinguish user and
    assistant history when the local-jsonl prompt-completion format is used.
    """
    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0].get("content") or ""

    parts = []
    for message in messages:
        role = message.get("role") or "unknown"
        content = message.get("content") or ""
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def build_row(example: dict[str, Any]) -> dict[str, str] | None:
    """Convert one Tulu conversation into an ``input``/``output`` row."""
    messages = example.get("messages") or []
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    assistant_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "assistant"),
        None,
    )
    if assistant_index is None or assistant_index == 0:
        return None

    completion = messages[assistant_index].get("content") or ""
    prompt = format_prompt(messages[:assistant_index])
    if not prompt or not completion:
        return None

    return {"input": prompt, "output": completion}


def parquet_rows(input_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield rows from all parquet shards in deterministic filename order."""
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        columns = set(parquet.schema_arrow.names)
        missing = REQUIRED_COLUMNS - columns
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"{parquet_file} is missing required columns: {missing_columns}")

        for batch in parquet.iter_batches(use_threads=True):
            for row in batch.to_pylist():
                yield row


def convert(input_dir: Path, output_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    """Convert parquet shards under ``input_dir`` into ``output_dir/training.jsonl``."""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / OUTPUT_FILENAME
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}\n"
            "Use --overwrite if it should be replaced."
        )

    row_count = 0
    skipped_count = 0
    with output_file.open("w", encoding="utf-8", newline="\n") as destination:
        for row in parquet_rows(input_dir):
            converted = build_row(row)
            if converted is None:
                skipped_count += 1
                continue
            destination.write(json.dumps(converted, ensure_ascii=False))
            destination.write("\n")
            row_count += 1

    return row_count, skipped_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Tulu 3 SFT mixture parquet shards to newline-delimited JSON."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source directory containing parquet shards.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help=f"Output directory; writes {OUTPUT_FILENAME} inside it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output training.jsonl if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        row_count, skipped_count = convert(args.src, args.dst, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_file = args.dst.expanduser().resolve() / OUTPUT_FILENAME
    print(f"Wrote {row_count:,} prompt-completion records to {output_file}")
    if skipped_count:
        print(f"Skipped {skipped_count:,} rows without a valid prompt/completion pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
