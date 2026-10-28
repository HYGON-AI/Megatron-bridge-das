# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from typing import Optional, Tuple

import torch
from torch import Tensor

import megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention as _upstream
from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
    get_flash_mla_topk_alignment,
)

_flash_mla_sparse_fwd = None


def _ensure_flash_mla():
    """Lazily import the FlashMLA sparse-forward kernel.

    FlashMLA ships ``flash_mla_sparse_fwd`` with a multi-head-KV signature;
    :func:`_csa_fwd_flash_mla` below is a thin adapter that unbatches the
    DSA-shape inputs and pads ``TopK`` to the alignment expected by
    FlashMLA's SM90 / SM100 kernels.
    """
    global _flash_mla_sparse_fwd
    if _flash_mla_sparse_fwd is not None:
        return

    try:
        from flash_mla import flash_mla_sparse_fwd as _fwd
    except ImportError as e:
        raise ImportError(
            "FlashMLA is required for DSA sparse attention forward. "
            "Install from https://github.com/deepseek-ai/FlashMLA/tree/nv_dev "
            "so that `from flash_mla import flash_mla_sparse_fwd` succeeds."
        ) from e
    _flash_mla_sparse_fwd = _fwd


def _ensure_dsa_namespace():
    """Try cudnn first, fall back to hip_dsa. Writes to the UPSTREAM module's _DSA."""
    if _upstream._DSA is not None:
        return
    try:
        from cudnn import DSA as _ns
    except ImportError as cudnn_error:
        try:
            from hip_dsa import DSA as _ns
        except ImportError as hip_error:
            raise ImportError(
                "No DSA namespace available. Install nvidia-cudnn-frontend[cutedsl] or hip_dsa. "
                f"(cudnn: {cudnn_error!r}; hip: {hip_error!r})"
            ) from None
    _upstream._DSA = _ns 


def _csa_fwd_flash_mla(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """DSA-shaped adapter around :func:`flash_mla.flash_mla_sparse_fwd`.

    Accepts flat (unbatched) tensors with global indices; pads ``TopK`` to
    the GPU-specific alignment; returns ``(out, lse, lse_indexer)``.
    """
    assert not (
        indexer_topk > 0 and topk_length is not None
    ), "indexer_topk > 0 requires non-compact mode (topk_length must be None)"
    _ensure_flash_mla()

    _total_S_q, _H, _D = q.shape
    TopK = topk_idxs.shape[-1]
    topk_align = get_flash_mla_topk_alignment()
    TopK_padded = (TopK + topk_align - 1) // topk_align * topk_align
    if TopK_padded != TopK:
        pad_width = TopK_padded - TopK
        topk_idxs = torch.nn.functional.pad(topk_idxs, (0, pad_width), value=-1)

    kv_3d = kv.unsqueeze(1)  # (total_S_kv, 1, D)  h_kv=1
    indices = topk_idxs.unsqueeze(1)  # (total_S_q, 1, TopK_padded) h_kv=1

    with torch.cuda.nvtx.range("flash_mla_sparse_fwd"):
        try:
            res = _flash_mla_sparse_fwd(
                q,
                kv_3d,
                indices,
                softmax_scale,
                d_v=d_v,
                attn_sink=attn_sink,
                topk_length=topk_length,
                indexer_topk=indexer_topk,
            )

        except Exception as e:
            from hip_dsa import sparse_attn_forward, sparse_attn_fwd_indexer

            dsa_sparse_fwd = sparse_attn_fwd_indexer if indexer_topk > 0 else sparse_attn_forward
            kwargs = dict(
                q=q,
                kv=kv_3d,
                attn_sink=attn_sink,
                topk_idxs=topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
            )
            if indexer_topk > 0:
                kwargs["indexer_topk"] = indexer_topk
            res = dsa_sparse_fwd(**kwargs)

        if indexer_topk > 0:
            out, _max_logits, lse, lse_indexer = res
        else:
            if len(res) == 2:
                res = (res[0], None, res[1])
            out, _max_logits, lse = res
            lse_indexer = None

    if indexer_topk > 0:
        # When indexer_topk == total TopK, lse_indexer should equal lse but
        # the kernel may not snapshot correctly; fall back to lse.
        if indexer_topk >= TopK:
            return out, lse, lse.clone()
        return out, lse, lse_indexer
    return out, lse, None