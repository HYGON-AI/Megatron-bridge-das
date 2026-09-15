# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn a COCO-caption CSV plus local images into a Mantis-shaped ``--source-dir``.

``prepare_mantis_energon.py`` expects one directory per subset, each holding
``train-*.parquet`` rows with an ``images`` column of ``{"path": ...}`` records
resolved *relative to that subset directory* and a ``conversation`` column of
ChatML turns whose ``content`` is a plain string. This script produces exactly
that layout from the ModelScope ``coco_2014_caption`` CSV export, whose columns
are ``uniq_id,image_id,caption,image`` and whose images are downloaded locally
as ``{uniq_id}_{image_id}_{basename-of-url}``.

Usage::

    uv run python examples/qwen/prepare_coco_caption_source.py \
        --csv /data/coco/train2014.csv \
        --images-dir /data/coco/coco_images \
        --output-dir /data/coco-source

    uv run python 3rdparty/Megatron-Bridge/examples/models/qwen/qwen3_vl/prepare_mantis_energon.py \
        --source-dir /data/coco-source \
        --output-dir /data/coco-energon \
        --max-samples-per-tar 10000 \
        --validation-fraction 0.01

Rows whose image is absent are skipped and counted rather than aborting the run,
because the published CSVs routinely list more rows than were downloaded.
"""

import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)

_IMAGE_TAG = "<image>"
_DEFAULT_PROMPT = f"介绍一下图片内容{_IMAGE_TAG}"
_MULTI_CAPTION_SEPARATOR = "&&"
_DEFAULT_COLUMNS = ("uniq_id", "image_id", "caption", "image")


def _index_by_basename(images_dir: Path) -> dict[str, str]:
    """Map each image's URL basename to its on-disk filename.

    Local files are named ``{uniq_id}_{image_id}_{basename}``. Neither id
    contains an underscore -- even the ``&&``-joined ids of the validation
    export -- so splitting twice recovers the basename exactly.
    """
    index = {}
    for entry in os.scandir(images_dir):
        if not entry.is_file():
            continue
        parts = entry.name.split("_", 2)
        if len(parts) == 3:
            index.setdefault(parts[2], entry.name)
    return index


def _resolve_image(
    row_uniq: str,
    row_image_id: str,
    url: str,
    images_dir: Path,
    basename_index: dict[str, str],
) -> Path | None:
    """Return the local file for one CSV row, or ``None`` when it is missing."""
    basename = url.rsplit("/", 1)[-1]
    candidate = images_dir / f"{row_uniq}_{row_image_id}_{basename}"
    if candidate.is_file():
        return candidate
    fallback = basename_index.get(basename)
    return images_dir / fallback if fallback else None


def _captions(raw: str, mode: str) -> list[str]:
    """Split a possibly ``&&``-joined caption cell according to ``mode``."""
    if _MULTI_CAPTION_SEPARATOR not in raw:
        return [raw.strip()]
    parts = [part.strip() for part in raw.split(_MULTI_CAPTION_SEPARATOR) if part.strip()]
    if not parts:
        return []
    if mode == "first":
        return parts[:1]
    if mode == "all":
        return parts
    return []


def _materialize(source: Path, subset_dir: Path, mode: str) -> str:
    """Place one image under ``subset_dir`` and return its relative path."""
    if mode == "reference":
        return os.path.relpath(source, subset_dir).replace(os.sep, "/")

    destination = subset_dir / "images" / source.name
    # lexists, not exists: a re-run over a moved source leaves broken symlinks,
    # and those must be replaced rather than raising FileExistsError.
    if not os.path.lexists(destination):
        if mode == "copy":
            shutil.copy2(source, destination)
        else:
            destination.symlink_to(source.resolve())
    return f"images/{source.name}"


def build_subset(
    csv_path: str,
    images_dir: str,
    output_dir: str,
    *,
    subset_name: str | None = None,
    prompt: str = _DEFAULT_PROMPT,
    rows_per_parquet: int = 5000,
    image_mode: str = "symlink",
    multi_caption: str = "first",
    columns: tuple[str, str, str, str] = _DEFAULT_COLUMNS,
    has_header: bool = True,
    encoding: str = "utf-8",
    limit: int | None = None,
) -> dict[str, int]:
    """Write one Mantis-shaped subset directory from a COCO-caption CSV.

    Args:
        csv_path: ModelScope ``coco_2014_caption`` CSV export.
        images_dir: Directory of ``{uniq_id}_{image_id}_{basename}`` images.
        output_dir: ``--source-dir`` to create; the subset becomes a child of it.
        subset_name: Subset directory name; defaults to the CSV file stem.
        prompt: User turn, which must contain exactly one ``<image>`` tag.
        rows_per_parquet: Rows per ``train-*.parquet`` shard.
        image_mode: ``symlink``, ``copy``, or ``reference`` (relative path only).
        multi_caption: ``first`` or ``all`` for ``&&``-joined caption cells.
        columns: Names of the uniq-id, image-id, caption, and URL columns.
        has_header: Whether the CSV carries a header row.
        encoding: CSV text encoding.
        limit: Optional cap on CSV rows read, for smoke tests.

    Returns:
        Counts of written samples and of each skip reason.

    Raises:
        ValueError: If the prompt lacks exactly one ``<image>`` tag, or if
            ``rows_per_parquet`` is not positive.
        FileNotFoundError: If the CSV or images directory is missing.
        RuntimeError: If no row produced a usable sample.
    """
    if prompt.count(_IMAGE_TAG) != 1:
        raise ValueError(f"--prompt must contain exactly one {_IMAGE_TAG} tag, got: {prompt!r}")
    if rows_per_parquet <= 0:
        raise ValueError("rows_per_parquet must be greater than 0.")

    csv_file = Path(csv_path)
    images_path = Path(images_dir)
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    if not images_path.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    uniq_col, image_id_col, caption_col, url_col = columns
    read_kwargs = {"encoding": encoding}
    if has_header:
        read_kwargs["usecols"] = list(columns)
    else:
        read_kwargs["header"] = None
        read_kwargs["names"] = list(columns)
    if limit is not None:
        read_kwargs["nrows"] = limit
    # The published CSVs quote captions that contain commas and newlines, so a
    # real parser is required here; splitting on commas silently corrupts rows.
    dataframe = pd.read_csv(csv_file, **read_kwargs)

    subset_dir = Path(output_dir) / (subset_name or csv_file.stem)
    subset_dir.mkdir(parents=True, exist_ok=True)
    if image_mode != "reference":
        (subset_dir / "images").mkdir(exist_ok=True)

    basename_index = _index_by_basename(images_path)
    logger.info("Indexed %d local images under %s", len(basename_index), images_path)

    records: list[dict] = []
    stats = {"samples": 0, "rows": len(dataframe), "missing_image": 0, "empty_caption": 0}

    for row in tqdm(dataframe.itertuples(index=False), total=len(dataframe), desc=subset_dir.name, unit="row"):
        raw_caption = getattr(row, caption_col)
        url = getattr(row, url_col)
        if not isinstance(raw_caption, str) or not isinstance(url, str):
            stats["empty_caption"] += 1
            continue

        source = _resolve_image(
            str(getattr(row, uniq_col)), str(getattr(row, image_id_col)), url, images_path, basename_index
        )
        if source is None:
            stats["missing_image"] += 1
            continue

        captions = _captions(raw_caption, multi_caption)
        if not captions:
            stats["empty_caption"] += 1
            continue

        relative = _materialize(source, subset_dir, image_mode)
        for caption in captions:
            records.append(
                {
                    "images": [{"path": relative}],
                    "conversation": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": caption},
                    ],
                }
            )
            stats["samples"] += 1

    if not records:
        raise RuntimeError(f"No usable rows in {csv_file}; {stats['missing_image']} rows had no local image.")

    shards = [records[start : start + rows_per_parquet] for start in range(0, len(records), rows_per_parquet)]
    for index, shard in enumerate(shards):
        name = f"train-{index:05d}-of-{len(shards):05d}.parquet"
        pd.DataFrame(shard).to_parquet(subset_dir / name, index=False)
    logger.info("Wrote %d samples across %d parquet files in %s", stats["samples"], len(shards), subset_dir)
    return stats


def main() -> None:
    """Parse arguments and build one subset per CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True, metavar="PATH", help="Repeatable; one subset per CSV")
    parser.add_argument("--images-dir", required=True, help="Directory of {uniq_id}_{image_id}_{basename} images")
    parser.add_argument("--output-dir", required=True, help="--source-dir to create for prepare_mantis_energon.py")
    parser.add_argument("--subset-name", help="Subset directory name; only valid with a single --csv")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT, help=f"User turn containing one {_IMAGE_TAG} tag")
    parser.add_argument("--rows-per-parquet", type=int, default=5000, metavar="N")
    parser.add_argument("--image-mode", choices=("symlink", "copy", "reference"), default="symlink")
    parser.add_argument(
        "--multi-caption",
        choices=("first", "all"),
        default="first",
        help="How to treat '&&'-joined caption cells found in the validation export.",
    )
    parser.add_argument("--columns", nargs=4, default=list(_DEFAULT_COLUMNS), metavar=("UNIQ", "IMAGE_ID", "CAPTION", "URL"))
    parser.add_argument("--no-header", action="store_true", help="The CSV has no header row.")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--limit", type=int, help="Read at most N CSV rows, for smoke tests.")
    args = parser.parse_args()

    if args.subset_name and len(args.csv) > 1:
        parser.error("--subset-name is only valid with a single --csv")

    for csv_path in args.csv:
        stats = build_subset(
            csv_path,
            args.images_dir,
            args.output_dir,
            subset_name=args.subset_name,
            prompt=args.prompt,
            rows_per_parquet=args.rows_per_parquet,
            image_mode=args.image_mode,
            multi_caption=args.multi_caption,
            columns=tuple(args.columns),
            has_header=not args.no_header,
            encoding=args.encoding,
            limit=args.limit,
        )
        logger.info("%s -> %s", csv_path, stats)
    logger.info("Done. Pass --source-dir %s to prepare_mantis_energon.py", args.output_dir)


if __name__ == "__main__":
    main()
