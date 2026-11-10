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

"""Streaming importer for DeepSeek-V4.1 Engram weights.

Checkpoint layout, verified against ``DeepSeek-V4.1-Flash-1017`` safetensors headers::

    layers.{hf}.engram.embed.weight  F8_E4M3  [rows, 256]    rows = sum(table_sizes)
    layers.{hf}.engram.embed.scale   F8_E8M0  [rows, 8]      per-row 32-element K tiles
    layers.{hf}.engram.wkv.weight    F8_E4M3  [25600, 6144]  32x32 blocks
    layers.{hf}.engram.wkv.scale     F8_E8M0  [800, 192]
    layers.{hf}.engram.q_weight      BF16     [num_streams, hidden]
    layers.{hf}.engram.k_weight      BF16     [num_streams, hidden]

One layer's embedding table is ~98 GiB of F8 data inside a single safetensors shard, so it
must be streamed with ``safe_open(...).get_slice()`` row windows; materialising the tensor
(the default bridge loader does that) needs 95 GiB of host memory per rank.

Megatron side (see ``megatron/core/models/engram``):

* ``engram.embedding.tables[i].weight`` -- ``[local_rows, head_dim]``; table ``i`` owns rows
  ``[range_start, range_start + local_rows)`` of the flat HF tensor, split evenly over EP.
* ``engram.key_projection.weight``  -- ``[num_streams * hidden, total_memory_dim]`` (wkv top).
* ``engram.value_projection.weight``-- ``[hidden, total_memory_dim]`` (wkv bottom).
* ``engram.query_norm.weight`` / ``key_norm.weight`` -- ``[num_streams * hidden]``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

try:  # safetensors is a hard dependency of the bridge, but keep the import lazy-friendly.
    from safetensors import safe_open
except ImportError:  # pragma: no cover - only hit in stripped environments
    safe_open = None


# 4M rows of 256 fp8 bytes == 1 GiB per streaming chunk.
DEFAULT_CHUNK_ROWS = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Checkpoint access
# ---------------------------------------------------------------------------
def resolve_weight_map(model_dir: str) -> Tuple[str, Dict[str, str]]:
    """Return ``(root_dir, {tensor_name: shard_file})`` for a HF checkpoint directory."""
    root = os.path.abspath(os.path.expanduser(str(model_dir)))
    index_files = sorted(
        name for name in os.listdir(root) if name.endswith(".safetensors.index.json")
    )
    if index_files:
        with open(os.path.join(root, index_files[0]), "r", encoding="utf-8") as handle:
            return root, json.load(handle)["weight_map"]

    single = [name for name in os.listdir(root) if name.endswith(".safetensors")]
    if len(single) == 1:
        return root, {name: single[0] for name in single}
    raise FileNotFoundError(f"no safetensors index found under {root}")


def read_tensor(root: str, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    """Load a whole tensor (only use for small tensors such as wkv / norms)."""
    if safe_open is None:
        raise RuntimeError("safetensors is required to import Engram weights")
    with safe_open(os.path.join(root, weight_map[key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def read_slice(
    root: str, weight_map: Dict[str, str], key: str, start: int, end: int
) -> torch.Tensor:
    """Load ``[start:end)`` rows of a 2-D tensor without materialising the rest."""
    if safe_open is None:
        raise RuntimeError("safetensors is required to import Engram weights")
    with safe_open(os.path.join(root, weight_map[key]), framework="pt", device="cpu") as handle:
        return handle.get_slice(key)[start:end]


def has_engram_tensors(weight_map: Dict[str, str]) -> bool:
    return any(".engram." in name for name in weight_map)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def dequantize_fp8(weight: torch.Tensor, scale: Optional[torch.Tensor]) -> torch.Tensor:
    """Dequantise ``F8_E4M3`` weights whose ``F8_E8M0`` scale uses one of two layouts.

    * per-row K tiles -- ``scale.shape[0] == weight.shape[0]`` with
      ``weight.shape[1] == scale.shape[1] * block`` (Engram embedding tables:
      ``[rows, 256]`` weight and ``[rows, 8]`` scale, block 32);
    * square 2-D blocks -- ``scale.shape == ceil(weight.shape / block)``
      (V4.1 attention / wkv: block 32, not the 128 V4 used).
    """
    if scale is None:
        return weight.float()

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"expected 2-D weight/scale, got {tuple(weight.shape)} / {tuple(scale.shape)}"
        )

    weight_f = weight.float()
    scale_f = scale.float()

    if scale_f.shape[0] == weight_f.shape[0]:
        if weight_f.shape[1] % scale_f.shape[1] != 0:
            raise ValueError(
                f"weight width {weight_f.shape[1]} is not a multiple of the scale width "
                f"{scale_f.shape[1]}"
            )
        block = weight_f.shape[1] // scale_f.shape[1]
        return (weight_f.unflatten(-1, (-1, block)) * scale_f.unsqueeze(-1)).flatten(-2)

    if weight_f.shape[0] % scale_f.shape[0] or weight_f.shape[1] % scale_f.shape[1]:
        raise ValueError(
            f"scale shape {tuple(scale_f.shape)} is incompatible with weight shape "
            f"{tuple(weight_f.shape)}"
        )
    row_block = weight_f.shape[0] // scale_f.shape[0]
    col_block = weight_f.shape[1] // scale_f.shape[1]
    if row_block != col_block:
        raise ValueError(
            f"non-square FP8 blocks are not supported: {row_block}x{col_block}"
        )
    dequantized = weight_f.unflatten(0, (-1, row_block)).unflatten(2, (-1, col_block))
    dequantized = dequantized * scale_f[:, None, :, None]
    return dequantized.flatten(2).flatten(0, 1)


def table_row_ranges(
    table_sizes: Sequence[int], ep_size: int, ep_rank: int
) -> List[Tuple[int, int]]:
    """Return ``(hf_row_start, local_rows)`` for every prime table on this EP rank.

    Megatron splits each table contiguously and shares the remainder over the first
    ``table_size % ep_size`` ranks, which is what ``EPShardedEmbeddingTable`` allocates.
    """
    if ep_size < 1:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    ranges: List[Tuple[int, int]] = []
    offset = 0
    for size in table_sizes:
        base, remainder = divmod(int(size), ep_size)
        local_rows = base + (1 if ep_rank < remainder else 0)
        # First ``remainder`` ranks carry one extra row, so their start is shifted.
        local_start = base * ep_rank + min(ep_rank, remainder)
        ranges.append((offset + local_start, local_rows))
        offset += int(size)
    return ranges


def split_wkv(
    wkv: torch.Tensor, key_rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split the fused Engram ``wkv`` into ``(key_projection, value_projection)`` weights."""
    if key_rows <= 0 or key_rows >= wkv.shape[0]:
        raise ValueError(f"key_rows={key_rows} is outside wkv rows {wkv.shape[0]}")
    return wkv[:key_rows], wkv[key_rows:]


def iter_table_chunks(
    root: str,
    weight_map: Dict[str, str],
    weight_key: str,
    scale_key: Optional[str],
    hf_row_start: int,
    local_rows: int,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Iterator[Tuple[int, torch.Tensor]]:
    """Yield ``(local_offset, dequantised_rows)`` windows for one EP-local table shard."""
    for local_offset in range(0, local_rows, chunk_rows):
        count = min(chunk_rows, local_rows - local_offset)
        absolute_start = hf_row_start + local_offset
        rows = read_slice(root, weight_map, weight_key, absolute_start, absolute_start + count)
        scales = (
            read_slice(root, weight_map, scale_key, absolute_start, absolute_start + count)
            if scale_key is not None
            else None
        )
        yield local_offset, dequantize_fp8(rows, scales)

def tensor_shape(root: str, weight_map: Dict[str, str], key: str) -> Tuple[int, ...]:
    """Return a tensor's shape from the safetensors header without reading any data."""
    if safe_open is None:
        raise RuntimeError("safetensors is required to import Engram weights")
    with safe_open(os.path.join(root, weight_map[key]), framework="pt", device="cpu") as handle:
        return tuple(handle.get_slice(key).get_shape())
