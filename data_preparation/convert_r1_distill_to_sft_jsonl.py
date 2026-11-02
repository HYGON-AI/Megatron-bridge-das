#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Convert Chinese-DeepSeek-R1-Distill-SFT into local-jsonl SFT splits.

Reads an Alpaca-style jsonl (fields: ``instruction``, ``input``, ``output``,
plus metadata that is dropped) and writes three prompt-completion files that
Megatron-Bridge's ``--dataset local-jsonl`` preset consumes:

    <dst>/training.jsonl
    <dst>/validation.jsonl
    <dst>/test.jsonl

Each output row is ``{"input": <prompt>, "output": <completion>}`` — the field
names required by ``PromptCompletionSFTPreprocessingConfig`` (prompt_column=
"input", completion_column="output"), NOT the source dataset's fields.

The source ``output`` includes ``<think>...</think>`` reasoning traces; we
keep them verbatim so the fine-tuned model learns the R1-style CoT format.
"""

import argparse
import json
import random
from pathlib import Path


def build_row(example: dict) -> dict | None:
    """Alpaca (instruction, input, output) -> prompt-completion (input, output)."""
    instruction = (example.get("instruction") or "").strip()
    extra_input = (example.get("input") or "").strip()
    completion = example.get("output") or ""
    if not instruction or not completion:
        return None
    prompt = instruction if not extra_input else f"{instruction}\n{extra_input}"
    return {"input": prompt, "output": completion}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=None, help="Source jsonl path.")
    parser.add_argument("--dst", default=None, help="Output directory.")
    parser.add_argument(
        "--val-ratio", type=float, default=0.01, help="Validation fraction (default 1%)."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.01, help="Test fraction (default 1%)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5678,
        help="Shuffle seed; 5678 matches GPTSFTDatasetConfig's default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_ratio + args.test_ratio >= 1.0:
        raise SystemExit("val-ratio + test-ratio must be < 1.0")

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Reading {src}")
    raw_lines: list[str] = []
    with src.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if line:
                raw_lines.append(line)
            if line_no % 20000 == 0:
                print(f"  read {line_no} lines")
    total = len(raw_lines)
    print(f"Total non-empty lines: {total}")

    rng = random.Random(args.seed)
    rng.shuffle(raw_lines)

    n_val = int(total * args.val_ratio)
    n_test = int(total * args.test_ratio)
    n_train = total - n_val - n_test
    splits = {
        "training.jsonl": raw_lines[:n_train],
        "validation.jsonl": raw_lines[n_train : n_train + n_val],
        "test.jsonl": raw_lines[n_train + n_val :],
    }
    print(f"Split plan: train={n_train} val={n_val} test={n_test} (seed={args.seed})")

    total_ok = 0
    total_skip = 0
    for name, chunk in splits.items():
        out_path = dst / name
        ok = 0
        skip = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for line in chunk:
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    skip += 1
                    continue
                row = build_row(example)
                if row is None:
                    skip += 1
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok += 1
        total_ok += ok
        total_skip += skip
        print(f"  wrote {out_path}  ok={ok}  skipped={skip}")

    print(f"Done. wrote={total_ok} skipped={total_skip} into {dst}")


if __name__ == "__main__":
    main()
