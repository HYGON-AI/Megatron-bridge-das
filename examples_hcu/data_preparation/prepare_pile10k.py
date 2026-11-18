# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Build a Megatron indexed dataset (``.bin``/``.idx``) from a local Pile sample.

The DeepSeek-V4-Flash GB200 pretrain card feeds ``the-pile/my-gpt3_08_text_document``
into ``--dataset megatron-indexed``. That name decomposes as:

    my-gpt3      GPT-2 BPE tokenized text (vocab 50257, so uint16 token ids)
    _08          shard index; the number is just a naming convention
    _text_document   suffix appended by preprocess_data.py (level=document)

``NeelNanda/pile-10k`` is the smallest public stand-in for The Pile: one parquet
shard (33 MB, 10k documents, ``text`` column). This script turns a *local* copy of
it into that indexed dataset; it never downloads anything.

Two stages, both idempotent:

1. ``jsonl``      stream the parquet shard(s) and write one ``{"text": ...}`` object
                  per non-blank row (one row = one document).
2. ``preprocess`` run ``3rdparty/Megatron-LM/tools/preprocess_data.py`` with
                  ``--append-eod`` and the chosen tokenizer, then report the
                  document/token counts parsed out of the ``.idx`` header.

Tokenizers: for a card-faithful ``my-gpt3`` prefix use ``GPT2BPETokenizer`` with the
GPT-2 ``vocab.json`` + ``merges.txt`` (``openai-community/gpt2``). Any HF tokenizer
directory works too, but then the token ids differ from the card's GPT-2 BPE ones.

Typical use (run inside the training container: needs numpy/pyarrow and a GPU stack
for ``preprocess_data.py``)::

    python examples_hcu/data_preparation/prepare_pile10k.py --stage all \
        --parquet       /data1/datasets/pile-10k/data/train-00000-of-00001-4746b8785c874cc7.parquet \
        --output-dir    /data1/datasets/the-pile \
        --vocab-file    /data1/datasets/the-pile/bpe/vocab.json \
        --merge-file    /data1/datasets/the-pile/bpe/merges.txt \
        --workers 32

Produces ``/data1/datasets/the-pile/my-gpt3_08_text_document.{bin,idx}``, which the
recipe consumes as::

    'dataset.blend=[["/data1/datasets/the-pile/my-gpt3_08_text_document"],null]'
    dataset.path_to_cache=/data1/datasets/the-pile/cache
"""

from __future__ import annotations

import argparse
import json
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path

try:  # imported eagerly when available, checked lazily so --help works anywhere
    import numpy
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only without dependencies
    numpy = None
    pq = None


DEFAULT_TEXT_COLUMN = "text"
STAGES = ("jsonl", "preprocess")


def require_data_libs() -> None:
    """Fail with an actionable message when numpy/pyarrow are unavailable."""
    if numpy is None or pq is None:
        raise SystemExit(
            "This program requires numpy and pyarrow. Install them with: pip install numpy pyarrow"
        )


def log(message: str) -> None:
    print(f"[prepare-pile10k] {message}", flush=True)


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _shell_quote(value: str) -> str:
    """Quote one argv element for the ``bash -lc`` wrapper."""
    return shlex.quote(str(value))


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


def collect_parquet_files(target: Path) -> list[Path]:
    """Return the parquet shards to read, or explain how to fix --parquet."""
    if target.is_file():
        if target.suffix != ".parquet":
            raise SystemExit(f"--parquet is not a parquet file: {target}")
        return [target]
    if not target.is_dir():
        raise SystemExit(
            f"--parquet does not exist: {target}\n"
            "Point it at the downloaded pile-10k parquet file (or at the directory holding it)."
        )
    shards = sorted(target.rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"No *.parquet under {target}")
    return shards


def stage_jsonl(args: argparse.Namespace) -> Path:
    """Convert every non-blank parquet row into one JSONL document."""
    output = Path(args.jsonl_path)
    if output.is_file() and output.stat().st_size > 0 and not args.overwrite:
        log(f"jsonl: present ({human(output.stat().st_size)}), skipped")
        return output

    shards = collect_parquet_files(Path(args.parquet))
    require_data_libs()
    log(f"jsonl: reading {len(shards)} shard(s)")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    documents = 0
    characters = 0
    started = time.time()

    with temporary.open("w", encoding="utf-8") as handle:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            columns = parquet.schema_arrow.names
            if args.text_column not in columns:
                raise SystemExit(
                    f"{shard} has no '{args.text_column}' column; found {columns}"
                )
            for batch in parquet.iter_batches(batch_size=4096, columns=[args.text_column]):
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
    log(
        f"jsonl: {documents} documents, {characters} characters, "
        f"{human(output.stat().st_size)} on disk in {time.time() - started:.1f}s"
    )
    return output


def build_tokenizer_args(args: argparse.Namespace) -> list[str]:
    """Translate the tokenizer selection into preprocess_data.py flags."""
    flags = ["--tokenizer-type", args.tokenizer_type]
    if args.tokenizer_type == "GPT2BPETokenizer":
        if not args.vocab_file or not args.merge_file:
            raise SystemExit(
                "GPT2BPETokenizer needs --vocab-file and --merge-file "
                "(openai-community/gpt2 vocab.json / merges.txt)."
            )
        for label, value in (("--vocab-file", args.vocab_file), ("--merge-file", args.merge_file)):
            path = Path(value).expanduser()
            if not path.is_file():
                raise SystemExit(f"{label} does not exist: {path}")
            flags += [label, str(path)]
        # This Megatron version routes GPT2BPETokenizer through MegatronTokenizer, which
        # asserts that a metadata.json sits next to the tokenizer. Without it the worker
        # dies and preprocess_data.py blocks forever on its result queue.
        metadata = Path(args.tokenizer_metadata).expanduser() if args.tokenizer_metadata else Path(
            str(args.vocab_file)
        ).expanduser().parent / "metadata.json"
        if not metadata.is_file():
            raise SystemExit(
                f"GPT2BPETokenizer requires a tokenizer metadata file, but {metadata} is missing.\n"
                "In this Megatron version GPT2BPETokenizer goes through MegatronTokenizer, which\n"
                "asserts metadata.json exists; the worker then dies and the parent hangs on q.get().\n"
                "Simplest fix: --tokenizer-type HuggingFaceTokenizer --tokenizer-model <dir with\n"
                "tokenizer.json> (same GPT-2 BPE ids). Or generate the metadata with\n"
                "MegatronTokenizer.write_metadata() and pass --tokenizer-metadata."
            )
        flags += ["--tokenizer-metadata", str(metadata)]
    else:
        if not args.tokenizer_model:
            raise SystemExit(f"--tokenizer-type {args.tokenizer_type} needs --tokenizer-model.")
        model = Path(args.tokenizer_model).expanduser()
        flags += ["--tokenizer-model", str(model)]
        if model.is_dir():
            present = [n for n in ("tokenizer.json", "tokenizer.model", "vocab.json") if (model / n).is_file()]
            if not present:
                log(f"warning: {model} has no tokenizer.json/tokenizer.model/vocab.json")
    return flags


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
        args.text_column,
        "--output-prefix",
        tool_prefix,
        "--workers",
        str(args.workers),
    ]
    command += build_tokenizer_args(args)
    if args.append_eod:
        command.append("--append-eod")
    if args.partitions and args.partitions > 1:
        command += ["--partitions", str(args.partitions)]

    env_script = Path(args.env_script) if args.env_script else None
    if env_script is not None and env_script.is_file():
        # preprocess_data.py initialises the GPU stack (torch.cuda) at startup, which on
        # this cluster only works after the DTK environment script has been sourced.
        command = [
            "bash",
            "-lc",
            "source {} >/dev/null 2>&1; exec {}".format(
                env_script, " ".join(_shell_quote(part) for part in command)
            ),
        ]

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

    Layout written by ``megatron.core.datasets.indexed_dataset``::

        b"MMIDIDX\\x00\\x00"  magic                 (9 bytes)
        <Q version / <B dtype code / <Q seq count / <Q doc count
        int32 x seq_count   sequence lengths
        int64 x seq_count   sequence pointers
        int64 x doc_count   document end indices
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
    dtype = {2: "uint16", 4: "int32", 8: "int64"}.get(dtype_code, f"{element} B/token")
    log(
        f"indexed dataset: {sequences} documents, {tokens} tokens, "
        f"bin {human(bin_bytes)} ({dtype}), idx {human(idx_path.stat().st_size)}"
    )
    log(f"  header: version {version}, dtype code {dtype_code}, {sequences}/{documents} seq/doc")
    log(f"  prefix: {prefix}")
    if sequences:
        log(
            f"  document length: min {int(lengths.min())} / mean {tokens / sequences:.1f} / "
            f"max {int(lengths.max())} tokens"
        )
    return tokens


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local pile-10k parquet into my-gpt3_<shard>_text_document.{bin,idx}. "
        "Nothing is downloaded; point --parquet at the file you already have.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all", help="Which stage to run.")
    parser.add_argument("--parquet", required=True, help="Parquet file or directory with *.parquet shards.")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Column holding the raw text.")
    parser.add_argument(
        "--output-dir",
        default="/data1/datasets/the-pile",
        help="Directory that receives the JSONL and the .bin/.idx pair.",
    )
    parser.add_argument(
        "--shard",
        default="08",
        help="Shard number used in the default prefix name (my-gpt3_<shard>_text_document).",
    )
    parser.add_argument("--prefix", default=None, help="Full output prefix; overrides --output-dir/--shard.")
    parser.add_argument(
        "--jsonl-path",
        default=None,
        help="Intermediate JSONL path; defaults to <output-dir>/pile-10k.jsonl.",
    )
    parser.add_argument(
        "--tokenizer-type",
        default="HuggingFaceTokenizer",
        help="preprocess_data.py tokenizer backend. HuggingFaceTokenizer on a local GPT-2 dir "
        "produces the same token ids as GPT2BPETokenizer and needs no metadata.json.",
    )
    parser.add_argument("--vocab-file", default=None, help="GPT-2 vocab.json (GPT2BPETokenizer only).")
    parser.add_argument("--merge-file", default=None, help="GPT-2 merges.txt (GPT2BPETokenizer only).")
    parser.add_argument(
        "--tokenizer-model",
        default="/data1/models/gpt2",
        help="Tokenizer directory for HuggingFaceTokenizer (GPT-2 by default).",
    )
    parser.add_argument(
        "--tokenizer-metadata",
        default=None,
        help="metadata.json for MegatronTokenizer-based tokenizers (auto-detected next to --vocab-file).",
    )
    parser.add_argument("--workers", type=int, default=32, help="preprocess_data.py worker count.")
    parser.add_argument("--partitions", type=int, default=1, help="preprocess_data.py partition count.")
    parser.add_argument(
        "--append-eod",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the end-of-document token after each document.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Stop after this many documents (0 = all); handy for smoke tests.",
    )
    parser.add_argument("--python", default=sys.executable, help="Interpreter for preprocess_data.py.")
    parser.add_argument(
        "--env-script",
        default="/opt/dtk/env.sh",
        help="Sourced before preprocess_data.py (GPU stack init); '' disables.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs that already exist.")
    parser.add_argument(
        "--preprocess-script",
        default=None,
        help="Path to Megatron-LM tools/preprocess_data.py (auto-detected by default).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir = str(output_dir)
    args.jsonl_path = str(
        Path(args.jsonl_path).expanduser().resolve()
        if args.jsonl_path
        else output_dir / "pile-10k.jsonl"
    )
    args.prefix = str(
        Path(args.prefix).expanduser().resolve()
        if args.prefix
        else output_dir / f"my-gpt3_{args.shard}_text_document"
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log(f"stage: {args.stage}   parquet: {args.parquet}   prefix: {args.prefix}")

    if args.stage in ("all", "jsonl"):
        stage_jsonl(args)
    if args.stage in ("all", "preprocess"):
        prefix = stage_preprocess(args)
        report_indexed_dataset(prefix)

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
