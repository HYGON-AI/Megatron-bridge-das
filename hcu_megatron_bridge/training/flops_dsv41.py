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

"""DSv4.1 ``csa_compress_ratios`` support for the bridge FLOPs estimator.

``megatron.bridge.training.utils.flop_utils.num_floating_point_operations`` models the
DSv4 ratio family {0, 4, 128} only: 0 is pure sliding window, 4 a top-k CSA compressor
plus indexer, 128 a 128x compressor. DeepSeek-V4.1 additionally uses

* ``compress_ratio == 1`` -- no compression at all: the compressor only applies the
  ``linear_wkv`` projection (``CSA2Compressor._forward_sbhd``) and attention runs over
  the full causal context;
* ``compress_ratio == 2`` -- the r2 gated pooling path (``_pool_r2``): the compressed
  sequence is ``seq_len / 2`` long.

With those layers in ``csa_compress_ratios`` the upstream estimator raises
``ValueError: csa_compress_ratios contains unsupported values: [2]. Only 0, 4, and 128
are supported.`` This patch keeps the upstream maths untouched for {0, 4, 128} and adds
the missing layers itself:

1. call the upstream estimator with the 1/2 layers demoted to the window-only bucket
   (ratio 0), which is what the estimator can soak up;
2. add back the compressed part of those layers as
   ``seqlen_sum * 3 * 2 * (heads * avg_comp_r * v_head_dim * 2 + compressor + indexer)``
   with ``avg_comp_r = (core_attn_seq_factor // r) / 2``, i.e. exactly the causal
   average the upstream code already uses for ratio 128.

The window term of the demoted bucket stays in the upstream sum, so ratio-1 layers are
over-estimated by at most ``window / (seq_len / 2)`` (about 3% at 8k). That is well
inside the accuracy of the estimator itself and keeps the cost continuous instead of
dropping 38 of the 40 layers to window-only attention.
"""

import copy
import logging

from megatron.bridge.training.utils import flop_utils

logger = logging.getLogger(__name__)

# Captured at import time, before ``MegatronPatchesManager.apply_patches`` swaps the
# module attribute, so the wrapped call always reaches the pristine upstream function.
_original_num_floating_point_operations = flop_utils.num_floating_point_operations

# Ratios the upstream estimator already understands (with dedicated formulas).
_UPSTREAM_RATIOS = frozenset({0, 4, 128})
# Ratios this patch adds: 1 = no compression, >1 = pooled to seq_len / ratio.
_PATCHED_RATIOS = frozenset({1, 2})

_warned = False


def _effective_seq_stats(model_cfg, batch_size, seqlen_sum, seqlen_squared_sum):
    """Mirror the upstream ``core_attn_seq_factor`` / ``seqlen_sum`` derivation."""
    if seqlen_sum is not None and batch_size > 0:
        effective_seq_length = seqlen_sum / batch_size
    else:
        effective_seq_length = model_cfg.seq_length
        seqlen_sum = batch_size * model_cfg.seq_length
    if seqlen_squared_sum is not None and seqlen_sum and seqlen_sum > 0:
        core_attn_seq_factor = seqlen_squared_sum / seqlen_sum
    else:
        core_attn_seq_factor = effective_seq_length
    return core_attn_seq_factor, seqlen_sum


def _compressor_indexer_flops(model_cfg, ratio, core_attn_seq_factor):
    """Per-layer attention + compressor + indexer cost for one ``ratio`` layer.

    Same units as the upstream DSv4 block: the returned value still needs the
    ``3 * 2`` (fwd + bwd, FMA) factor and the outer ``seqlen_sum`` multiplier.
    """
    num_heads = model_cfg.num_attention_heads
    v_head_dim = getattr(model_cfg, "v_head_dim", 64)
    hidden_size = model_cfg.hidden_size
    q_lora_rank = getattr(model_cfg, "q_lora_rank", None) or 0

    # Number of compressed tokens a query attends to on average (causal -> half of the
    # compressed sequence), matching upstream's ``avg_comp_128``.
    avg_comp = (core_attn_seq_factor // ratio) / 2
    sparse_attn = num_heads * avg_comp * v_head_dim * 2

    # The demoted bucket already paid for the sliding window, so only the projections
    # are added here: ``linear_wkv`` alone for ratio 1, ``linear_wkv`` + ``linear_wgate``
    # (the r2 gated pooling) for ratio >= 2.
    if ratio == 1:
        compressor = hidden_size * v_head_dim * 2
    else:
        compressor = hidden_size * (2 * v_head_dim) * 2

    indexer = 0
    idx_n_heads = getattr(model_cfg, "dsa_indexer_n_heads", None)
    idx_head_dim = getattr(model_cfg, "dsa_indexer_head_dim", None)
    if ratio >= 2 and idx_n_heads and idx_head_dim:
        compressed_len = int(core_attn_seq_factor) // ratio
        indexer = (
            hidden_size * (2 * idx_head_dim) * 2
            + q_lora_rank * idx_n_heads * idx_head_dim
            + hidden_size * idx_n_heads
            + idx_n_heads * idx_head_dim * compressed_len
        )
    return sparse_attn + compressor + indexer


def num_floating_point_operations(
    cfg,
    batch_size=1,
    seqlen_sum=None,
    seqlen_squared_sum=None,
    num_vision_patches=0,
    cross_seqlen_sum=None,
    cross_seqlen_product_sum=None,
):
    """``flop_utils.num_floating_point_operations`` with DSv4.1 ratios 1/2 supported."""
    global _warned

    model_cfg = getattr(cfg, "model", None)
    compress_ratios = getattr(model_cfg, "csa_compress_ratios", None) if model_cfg is not None else None
    is_dsv4 = getattr(model_cfg, "experimental_attention_variant", None) == "dsv4_hybrid"

    # A provider-supplied estimator owns its own maths (upstream returns it early and
    # never validates the ratio array), so do not add anything on top of it.
    has_custom_estimator = hasattr(model_cfg, "_get_num_floating_point_operations") or hasattr(
        model_cfg, "_get_num_floating_point_operations_with_runtime_stats"
    )

    if (
        has_custom_estimator
        or not is_dsv4
        or not compress_ratios
        or set(compress_ratios) <= _UPSTREAM_RATIOS
    ):
        return _original_num_floating_point_operations(
            cfg,
            batch_size=batch_size,
            seqlen_sum=seqlen_sum,
            seqlen_squared_sum=seqlen_squared_sum,
            num_vision_patches=num_vision_patches,
            cross_seqlen_sum=cross_seqlen_sum,
            cross_seqlen_product_sum=cross_seqlen_product_sum,
        )

    ratios = [int(ratio) for ratio in compress_ratios]
    extra_ratios = [ratio for ratio in ratios if ratio not in _UPSTREAM_RATIOS]
    if any(ratio not in _PATCHED_RATIOS for ratio in extra_ratios):
        logger.warning(
            "FLOPs: csa_compress_ratios %s contain ratios outside {0, 4, 128} that this "
            "patch does not model explicitly; they are costed as plain compressed "
            "attention with avg_comp = seq_len / (2 * ratio).",
            extra_ratios,
        )

    demoted = copy.copy(model_cfg)
    demoted.csa_compress_ratios = [
        0 if ratio in extra_ratios else ratio for ratio in ratios
    ]
    demoted_cfg = copy.copy(cfg)
    demoted_cfg.model = demoted
    base = _original_num_floating_point_operations(
        demoted_cfg,
        batch_size=batch_size,
        seqlen_sum=seqlen_sum,
        seqlen_squared_sum=seqlen_squared_sum,
        num_vision_patches=num_vision_patches,
        cross_seqlen_sum=cross_seqlen_sum,
        cross_seqlen_product_sum=cross_seqlen_product_sum,
    )

    core_attn_seq_factor, seqlen_total = _effective_seq_stats(
        model_cfg, batch_size, seqlen_sum, seqlen_squared_sum
    )
    extra_per_token = 0.0
    for ratio in extra_ratios:
        # ``3 * 2`` = forward + backward x FMA, the upstream self-attention convention.
        extra_per_token += 3 * 2 * _compressor_indexer_flops(model_cfg, ratio, core_attn_seq_factor)
    total = base + seqlen_total * extra_per_token

    if not _warned:
        _warned = True
        from collections import Counter

        logger.warning(
            "FLOPs: DSv4.1 csa_compress_ratios %s are not modelled by the bridge "
            "estimator; added compressed-attention terms for ratios %s "
            "(+%.3e FLOPs per token, ratio 1 = full causal, ratio 2 = seq_len / 2).",
            dict(sorted(Counter(ratios).items())),
            extra_ratios,
            extra_per_token,
        )
    return int(total)
