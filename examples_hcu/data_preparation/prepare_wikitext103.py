# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Build a Megatron indexed dataset (``.bin``/``.idx``) from WikiText-103.

WikiText-103 is distributed as the Hugging Face dataset ``Salesforce/wikitext``,
configuration ``wikitext-103-raw-v1``: four parquet shards in one directory
(``train-00000-of-00002.parquet``, ``train-00001-of-00002.parquet``,
``validation-*.parquet``, ``test-*.parquet``). Download them yourself, then point
``--parquet-dir`` at the directory that holds them.

Megatron pretraining does not read parquet; it reads an *indexed dataset* made of
two files sharing one prefix::

    <prefix>.bin    token ids, little-endian int32/uint16 (preprocess_data.py picks
                    the smallest dtype that fits the tokenizer vocabulary)
    <prefix>.idx    MMIDIDX header + one int32 length per document

This script runs the two stages that turn the parquet shards into that pair:

1. ``jsonl``      stream the parquet shards batch by batch and write one
                  ``{"text": ...}`` object per non-blank row. One row is one
                  document, which is the granularity Megatron indexes.
2. ``preprocess`` call ``3rdparty/Megatron-LM/tools/preprocess_data.py`` with the
                  target model's tokenizer and ``--append-eod`` so the document
                  boundary is explicit, then report the document/token counts
                  parsed out of the ``.idx`` header.

The tokenizer decides the token ids, so the output prefix is tagged with it:
``wikitext103_glm52_text_document`` for ``/data1/models/GLM-5.2``. Always
pretrain with the same tokenizer revision that produced the dataset; a mismatch
shifts every token id and is silent.

Both the intermediate ``wikitext103.jsonl`` and the final ``.bin``/``.idx`` pair
are written into ``--output-dir``; nothing is written next to the parquet shards.

Typical use (run inside the training container: needs numpy/pyarrow and a GPU
stack for ``preprocess_data.py``)::

    python examples_hcu/data_preparation/prepare_wikitext103.py \
        --parquet-dir /data1/datasets/wikitext/wikitext-103-raw-v1 \
        --output-dir  /data1/datasets/wikitext103 \
        --tokenizer-model /data1/models/GLM-5.2 \
        --workers 32

Every stage is idempotent: an existing JSONL or indexed dataset is reused unless
``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path

try:  # imported eagerly when available, checked lazily so --help works on a bare interpreter
    import numpy
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only without dependencies
    numpy = None
    pq = None


def require_data_libs() -> None:
    """Fail with an actionable message when numpy/pyarrow are unavailable."""
    if numpy is None or pq is None:
        raise SystemExit(
            "This program requires numpy and pyarrow. Install them with: pip install numpy pyarrow"
        )


DEFAULT_SUBSET = "wikitext-103-raw-v1"
EXPECTED_SHARDS = (
    "train-00000-of-00002.parquet",
    "train-00001-of-00002.parquet",
    "validation-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
)
TEXT_COLUMN = "text"
STAGES = ("jsonl", "preprocess")


def log(message: str) -> None:
    print(f"[prepare-wikitext103] {message}", flush=True)


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _shell_quote(value: str) -> str:
    """Quote one argv element for the ``bash -lc`` wrapper."""
    return shlex.quote(str(value))


def _missing_tokenizer_files(model: str) -> list[str]:
    """Return the tokenizer files a HuggingFaceTokenizer directory is missing."""
    directory = Path(model).expanduser()
    if not directory.is_dir():
        return []
    expected = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json", "vocab.txt")
    present = [name for name in expected if (directory / name).is_file()]
    return [] if present else list(expected)


def resolve_preprocess_script(explicit: str | None) -> Path:
    """Locate ``tools/preprocess_data.py`` inside the Megatron-LM checkout."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"--preprocess-script does not exist: {path}")
        return path

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "3rdparty" / "Megatron-LM" / "tools" / "preprocess_data.py",
        Path("/data1/Megatron-LM/tools/preprocess_data.py"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "Could not find tools/preprocess_data.py. Pass --preprocess-script <path>, "
        "or initialize the Megatron-LM submodule (git submodule update --init)."
    )


def check_parquet_input(parquet_dir: Path) -> list[Path]:
    """Return the training shards, or explain what the caller has to download."""
    if not parquet_dir.is_dir():
        raise SystemExit(
            f"--parquet-dir does not exist: {parquet_dir}\n"
            f"Download {DEFAULT_SUBSET} and place these files there:\n  "
            + "\n  ".join(EXPECTED_SHARDS)
        )
    shards = sorted(parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise SystemExit(
            f"No train-*.parquet under {parquet_dir}\n"
            f"Expected at least {EXPECTED_SHARDS[0]} and {EXPECTED_SHARDS[1]}."
        )
    return shards


def stage_jsonl(args: argparse.Namespace) -> Path:
    """Convert every non-blank parquet row into one JSONL document."""
    output = Path(args.jsonl_path)
    if output.is_file() and output.stat().st_size > 0 and not args.overwrite:
        log(f"jsonl: present ({human(output.stat().st_size)}), skipped")
        return output

    shards = check_parquet_input(Path(args.parquet_dir))
    require_data_libs()
    log(f"jsonl: reading {len(shards)} shard(s) from {args.parquet_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    documents = 0
    characters = 0
    started = time.time()

    with temporary.open("w", encoding="utf-8") as handle:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            if TEXT_COLUMN not in parquet.schema_arrow.names:
                raise SystemExit(f"{shard} has no '{TEXT_COLUMN}' column; found {parquet.schema_arrow.names}")
            for batch in parquet.iter_batches(batch_size=8192, columns=[TEXT_COLUMN]):
                for value in batch.column(0).to_pylist():
                    text = (value or "").strip()
                    if not text:
                        continue
                    handle.write(json.dumps({"text": text}, ensure_ascii=False))
                    handle.write("\n")
                    documents += 1
                    characters += len(text)
                    if args.max_documents and documents >= args.max_documents:
                        break
                if args.max_documents and documents >= args.max_documents:
                    break
            if args.max_documents and documents >= args.max_documents:
                break

    temporary.replace(output)
    elapsed = time.time() - started
    log(
        f"jsonl: {documents} documents, {characters} characters, "
        f"{human(output.stat().st_size)} on disk in {elapsed:.1f}s"
    )
    return output


def stage_preprocess(args: argparse.Namespace) -> Path:
    prefix = Path(args.prefix)
    bin_path = prefix.with_name(prefix.name + ".bin")
    idx_path = prefix.with_name(prefix.name + ".idx")

    if bin_path.is_file() and idx_path.is_file() and not args.overwrite:
        log(f"indexed dataset: present ({bin_path.name}, {idx_path.name}), skipped")
        return prefix

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.is_file():
        raise SystemExit(f"{jsonl_path} is missing; run --stage jsonl first.")

    script = resolve_preprocess_script(args.preprocess_script)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # preprocess_data.py names its output "{prefix}_{json_key}_{level}.{bin,idx}",
    # so strip the "_text_document" suffix from the prefix we hand to the tool.
    tool_prefix = str(prefix)
    if tool_prefix.endswith("_text_document"):
        tool_prefix = tool_prefix[: -len("_text_document")]
    else:
        log("warning: prefix does not end with '_text_document'; the tool will append it")

    command = [
        str(args.python),
        str(script),
        "--input",
        str(jsonl_path),
        "--json-keys",
        TEXT_COLUMN,
        "--tokenizer-type",
        args.tokenizer_type,
        "--tokenizer-model",
        str(args.tokenizer_model),
        "--output-prefix",
        tool_prefix,
        "--workers",
        str(args.workers),
    ]
    if args.append_eod:
        command.append("--append-eod")
    if args.split_sentences:
        command.append("--split-sentences")

    env_script = Path(args.env_script) if args.env_script else None
    if env_script is not None and env_script.is_file():
        # preprocess_data.py initialises the GPU stack (torch.cuda) at startup, which on
        # this cluster only works after the DTK environment script has been sourced.
        shell_command = "source {} >/dev/null 2>&1; exec {}".format(
            env_script, " ".join(_shell_quote(part) for part in command)
        )
        command = ["bash", "-lc", shell_command]

    log(f"preprocessing: {' '.join(command)}")
    started = time.time()
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"preprocess_data.py failed with exit code {completed.returncode}")
    if not (bin_path.is_file() and idx_path.is_file()):
        raise SystemExit(f"preprocess_data.py did not produce {bin_path} / {idx_path}")
    log(f"preprocessing finished in {time.time() - started:.1f}s")
    return prefix


def report_indexed_dataset(prefix: Path) -> int:
    """Print the document/token counts stored in the ``.idx`` file.

    The index layout written by ``megatron.core.datasets.indexed_dataset`` is::

        b"MMIDIDX\\x00\\x00"  magic                     (9 bytes)
        <Q version                                (1)
        <B token dtype code                        (1)
        <Q sequence count                          (1)
        <Q document count                          (1)
        int32 x sequence_count   sequence lengths
        int64 x sequence_count   sequence pointers
        int64 x document_count   document end indices
    """
    require_data_libs()

    bin_path = prefix.with_name(prefix.name + ".bin")
    idx_path = prefix.with_name(prefix.name + ".idx")

    with idx_path.open("rb") as handle:
        header = handle.read(34)
    if len(header) < 34 or not header.startswith(b"MMIDIDX"):
        raise SystemExit(f"{idx_path} is not a Megatron indexed dataset index file")

    version = struct.unpack_from("<Q", header, 9)[0]
    dtype_code = struct.unpack_from("<B", header, 17)[0]
    sequences = struct.unpack_from("<Q", header, 18)[0]
    documents = struct.unpack_from("<Q", header, 26)[0]

    lengths = numpy.fromfile(idx_path, dtype=numpy.int32, count=sequences, offset=34)
    tokens = int(lengths.sum())
    bin_bytes = bin_path.stat().st_size
    element = bin_bytes // tokens if tokens else 0
    dtype = {2: "uint16", 4: "int32", 8: "int64"}.get(dtype_code, f"code {dtype_code}")
    log(
        f"indexed dataset: {sequences} documents, {tokens} tokens, "
        f"bin {human(bin_bytes)} ({dtype}, {element} B/token), idx {human(idx_path.stat().st_size)}"
    )
    log(f"  header: version {version}, dtype code {dtype_code}, {sequences}/{documents} seq/doc")
    log(f"  prefix: {prefix}")
    if sequences:
        log(
            f"  document length: min {int(lengths.min())} / mean {tokens / sequences:.1f} / "
            f"max {int(lengths.max())} tokens"
        )
    return tokens


def derive_tag(tokenizer_model: str, explicit: str | None) -> str:
    """Derive the ``wikitext103_<tag>_text_document`` tag from the tokenizer name."""
    if explicit:
        return explicit
    name = Path(tokenizer_model).name.strip("/") or "model"
    lowered = name.lower()
    if lowered.startswith("glm"):
        return lowered.replace(".", "").replace("-", "").replace("_", "")
    return lowered.replace(".", "").replace("-", "_").replace("/", "_")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WikiText-103 parquet shards into a Megatron indexed dataset (.bin/.idx). "
        "The parquet shards must already be on disk; this script never downloads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=(*STAGES, "all"),
        default="all",
        help="Run one stage, or both in order.",
    )
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Directory holding the train-*.parquet shards that you downloaded.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data1/datasets/wikitext103",
        help="Directory that receives wikitext103.jsonl and the .bin/.idx pair.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="/data1/models/GLM-5.2",
        help="Local tokenizer directory (or hub id) used to tokenize the text.",
    )
    parser.add_argument(
        "--tokenizer-type",
        default="HuggingFaceTokenizer",
        help="Tokenizer backend understood by preprocess_data.py.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Suffix of the output prefix; defaults to the tokenizer name (e.g. glm52).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Full output prefix without .bin/.idx; overrides --output-dir/--tag.",
    )
    parser.add_argument("--workers", type=int, default=32, help="preprocess_data.py worker count.")
    parser.add_argument(
        "--append-eod",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the end-of-document token after each document.",
    )
    parser.add_argument(
        "--split-sentences",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let preprocess_data.py split documents into sentences (usually off).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Stop after this many documents (0 = all); useful for smoke tests.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter used to run preprocess_data.py (needs numpy/pyarrow).",
    )
    parser.add_argument(
        "--env-script",
        default="/opt/dtk/env.sh",
        help="Shell script sourced before preprocess_data.py runs (GPU stack init); '' disables.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs that already exist.")
    parser.add_argument(
        "--preprocess-script",
        default=None,
        help="Path to Megatron-LM tools/preprocess_data.py (auto-detected by default).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    tag = derive_tag(args.tokenizer_model, args.tag)
    args.parquet_dir = str(Path(args.parquet_dir).expanduser().resolve())
    args.output_dir = str(output_dir)
    args.jsonl_path = str(output_dir / "wikitext103.jsonl")
    args.prefix = str(
        Path(args.prefix).expanduser().resolve()
        if args.prefix
        else output_dir / f"wikitext103_{tag}_text_document"
    )
    args.prefix = str(
        Path(args.prefix).expanduser().resolve()
        if args.prefix
        else output_dir / f"wikitext103_{tag}_text_document"
    )

    if args.stage in ("all", "preprocess"):
        missing = _missing_tokenizer_files(str(args.tokenizer_model))
        if missing:
            log(f"warning: {args.tokenizer_model} has no tokenizer files (missing {', '.join(missing)})")
            log("         HuggingFaceTokenizer will fail; fetch tokenizer.json/tokenizer_config.json first")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log(f"stage: {args.stage}   parquet: {args.parquet_dir}   tokenizer: {args.tokenizer_model}")

    if args.stage in ("all", "jsonl"):
        stage_jsonl(args)
    if args.stage in ("all", "preprocess"):
        prefix = stage_preprocess(args)
        report_indexed_dataset(prefix)

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
