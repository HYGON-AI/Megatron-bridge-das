# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

import runpy
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "examples" / "models" / "qwen" / "qwen3_vl" / "prepare_coco_caption_source.py"

_HEADER = "uniq_id,image_id,caption,image\n"
_URL = "https://example.com/open_data/coco_2014_caption/train/COCO_train2014_000000199134.jpg"
_IMAGE_NAME = "490489_199134_COCO_train2014_000000199134.jpg"


def _load_module() -> dict:
    return runpy.run_path(str(SCRIPT))


def _make_source(tmp_path: Path, rows: str, *, header: str = _HEADER) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    (images / _IMAGE_NAME).write_bytes(b"test-jpeg-bytes")
    csv_path = tmp_path / "train2014.csv"
    csv_path.write_text(header + rows, encoding="utf-8")
    return csv_path, images


def test_build_subset_writes_mantis_shaped_parquet(tmp_path: Path):
    module = _load_module()
    csv_path, images = _make_source(tmp_path, f"490489,199134,a man on a stage.,{_URL}\n")

    stats = module["build_subset"](str(csv_path), str(images), str(tmp_path / "source"))

    assert stats == {"samples": 1, "rows": 1, "missing_image": 0, "empty_caption": 0}
    subset = tmp_path / "source" / "train2014"
    frame = pd.read_parquet(subset / "train-00000-of-00001.parquet")
    row = frame.iloc[0]

    # Mirrors prepare_mantis_energon.convert(): the path resolves relative to
    # the subset directory and every turn's content must be a plain string.
    assert (subset / row["images"][0]["path"]).read_bytes() == b"test-jpeg-bytes"
    conversation = [dict(turn) for turn in row["conversation"]]
    assert [turn["role"] for turn in conversation] == ["user", "assistant"]
    assert sum(turn["content"].count("<image>") for turn in conversation) == len(row["images"])
    assert conversation[1]["content"] == "a man on a stage."


def test_build_subset_skips_rows_without_a_local_image(tmp_path: Path):
    module = _load_module()
    absent = "https://example.com/open_data/coco_2014_caption/train/COCO_train2014_000000000001.jpg"
    csv_path, images = _make_source(
        tmp_path,
        f"490489,199134,a man on a stage.,{_URL}\n1,2,a missing picture.,{absent}\n",
    )

    stats = module["build_subset"](str(csv_path), str(images), str(tmp_path / "source"))

    assert stats["samples"] == 1
    assert stats["missing_image"] == 1


def test_build_subset_expands_double_ampersand_captions(tmp_path: Path):
    module = _load_module()
    csv_path, images = _make_source(tmp_path, f'490489,199134,"a dog.&&a dog on a rug.",{_URL}\n')

    first = module["build_subset"](str(csv_path), str(images), str(tmp_path / "first"))
    every = module["build_subset"](
        str(csv_path), str(images), str(tmp_path / "all"), multi_caption="all", subset_name="s"
    )

    assert first["samples"] == 1
    assert every["samples"] == 2
    captions = [
        dict(turn)["content"]
        for turn in pd.read_parquet(tmp_path / "all" / "s" / "train-00000-of-00001.parquet").iloc[1]["conversation"]
    ]
    assert captions[1] == "a dog on a rug."


def test_build_subset_rejects_a_prompt_without_one_image_tag(tmp_path: Path):
    module = _load_module()
    csv_path, images = _make_source(tmp_path, f"490489,199134,a man on a stage.,{_URL}\n")

    with pytest.raises(ValueError, match="exactly one <image>"):
        module["build_subset"](str(csv_path), str(images), str(tmp_path / "source"), prompt="no tag here")


def test_build_subset_reads_headerless_csv_and_shards_parquet(tmp_path: Path):
    module = _load_module()
    csv_path, images = _make_source(tmp_path, f"490489,199134,a man on a stage.,{_URL}\n", header="")

    stats = module["build_subset"](
        str(csv_path),
        str(images),
        str(tmp_path / "source"),
        has_header=False,
        multi_caption="all",
        rows_per_parquet=1,
        image_mode="copy",
    )

    assert stats["samples"] == 1
    subset = tmp_path / "source" / "train2014"
    assert sorted(path.name for path in subset.glob("*.parquet")) == ["train-00000-of-00001.parquet"]
    assert not (subset / "images" / _IMAGE_NAME).is_symlink()
