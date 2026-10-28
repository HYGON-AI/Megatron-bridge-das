# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Fused kernels for mHC (Manifold-Constrained Hyper-Connections).

Uses Hip and Triton and cuda.tile (cuTile) kernels when available, with PyTorch
reference implementations as fallback.  Reference (non-fused) implementations
live in ``megatron.core.transformer.hyper_connection`` and are used when fused
kernels are unavailable or when the ``use_fused_mhc`` config flag is False.

Four fused operations:
  - sinkhorn:            Sinkhorn-Knopp projection to doubly stochastic matrix
  - h_aggregate:         weighted n-stream -> 1-stream aggregation
  - h_post_bda:          fused H_res.T @ residual + H_post * (x + bias)
  - proj_rms_compute_h:  fused projection + RMS normalization + compute_h
"""

import os
from typing import Optional, Tuple

from torch import Tensor

import megatron.core.fusions.fused_mhc_kernels as _megatron_fused_mhc_kernels
from megatron.core.transformer.hyper_connection import (
    native_sinkhorn,
    native_h_aggregate,
    native_h_post_bda,
)
from megatron.core.fusions.fused_mhc_kernels import (
    _TRITON_AVAILABLE,
    _get_triton_sinkhorn,
    _get_triton_h_aggregate_fwd,
    _get_triton_h_post_bda_fwd,
    _get_triton_h_post_bda_bwd,
    _raise_mhc_backend_validation_error,
    _torch_proj_rms_compute_h,
    is_cutile_available,
    FusedHAggregate,
    FusedHAggregateInto,
    FusedHPostBDA,
)
if is_cutile_available():
    from megatron.core.fusions.fused_mhc_kernels import (
        CutileSinkhornKnopp,
        CutileProjRmsComputeH,
    )
else:
    CutileSinkhornKnopp = None
    CutileProjRmsComputeH = None


def _forced_backend() -> Tuple[str, Optional[Exception]]:
    value = os.getenv("MHC_FORCE_BACKEND", "auto").strip().lower()
    value = value.replace("-", "_").replace("+", "_")
    aliases = {
        "auto": "auto",
        "mixed": "auto",
        "default": "auto",
        "native": "native",
        "torch": "native",
        "pytorch": "native",
        "none": "native",
        "triton": "triton",
        "triton_native": "triton",
        "cutile": "cutile",
        "cu_tile": "cutile",
        "cuda_tile": "cutile",
        "hip": "hip",
    }
    if value not in aliases:
        valid = ", ".join(sorted(aliases))
        return "auto", ValueError(
            f"Unsupported MHC_FORCE_BACKEND={value!r}; expected one of: {valid}"
        )
    return aliases[value], None


# ---------------------------------------------------------------------------
# Check Hip availability
# ---------------------------------------------------------------------------
_HIP_AVAILABLE = False
try:
    from fused_mhc import (
        FusedSinkhorn as HipFusedSinkhorn,
        FusedHAggregate as HipFusedHAggregate,
        FusedHAggregateInto as HipFusedHAggregateInto,
        FusedHPostBDA as HipFusedHPostBDA,
        FusedProjRmsComputeH as HipFusedProjRmsComputeH,
    )
    _HIP_AVAILABLE = True
except ImportError:
    pass


_MHC_FORCED_BACKEND, _MHC_BACKEND_VALIDATION_ERROR = _forced_backend()


if _MHC_FORCED_BACKEND == "hip":
    if not _HIP_AVAILABLE:
        _MHC_BACKEND_VALIDATION_ERROR = RuntimeError(
            "MHC_FORCE_BACKEND=hip was requested, but the fused_mhc is not available"
        )
    _megatron_fused_mhc_kernels._MHC_FORCED_BACKEND = "hip"
    _megatron_fused_mhc_kernels._MHC_BACKEND_VALIDATION_ERROR = _MHC_BACKEND_VALIDATION_ERROR
    _megatron_fused_mhc_kernels._TRITON_AVAILABLE = False
    _megatron_fused_mhc_kernels._CUTILE_AVAILABLE = False
    _megatron_fused_mhc_kernels._CUTILE_EXPERIMENTAL_AVAILABLE = False
    _TRITON_AVAILABLE = False
elif _MHC_BACKEND_VALIDATION_ERROR is not None:
    _megatron_fused_mhc_kernels._MHC_BACKEND_VALIDATION_ERROR = _MHC_BACKEND_VALIDATION_ERROR
else:
    _megatron_fused_mhc_kernels._MHC_BACKEND_VALIDATION_ERROR = None
    if _MHC_FORCED_BACKEND in ("native", "triton", "cutile"):
        _HIP_AVAILABLE = False


def is_hip_available() -> bool:
    """Return True if hip fused kernels are enabled."""
    return _HIP_AVAILABLE


# ============================================================================
# Unified public dispatch
# ============================================================================
# The public fused API chooses the fastest validated backend per operation:
#
#   sinkhorn fwd/bwd:       Hip -> Triton -> cuTile -> torch
#   h_post_bda fwd/bwd:     Hip -> Triton -> cuTile -> torch
#   h_aggregate fwd:        Hip -> Triton -> cuTile -> torch
#   h_aggregate bwd:        Hip -> cuTile -> torch
#   proj_rms/proj_rms_compute_h: Hip -> cuTile -> torch
#
# Runtime CUDA launch failures are intentionally not swallowed; after such an
# error the CUDA context may not be safely reusable for fallback work.
# ============================================================================


def _select_hip_triton_cutile_native(triton_impl) -> str:
    if is_hip_available():
        return "hip"
    if triton_impl is not None:
        return "triton"
    if is_cutile_available():
        return "cutile"
    return "native"


def _mhc_backend_status() -> Tuple[str, bool]:
    """Return backend description and whether every backend is native."""
    sinkhorn = _select_hip_triton_cutile_native(_get_triton_sinkhorn())
    h_aggregate_fwd = _select_hip_triton_cutile_native(_get_triton_h_aggregate_fwd())
    h_aggregate_bwd = _select_hip_triton_cutile_native(None)
    h_post_bda_fwd = _select_hip_triton_cutile_native(_get_triton_h_post_bda_fwd())
    h_post_bda_bwd = _select_hip_triton_cutile_native(_get_triton_h_post_bda_bwd())
    proj_rms_compute_h = _select_hip_triton_cutile_native(None)
    selected = (
        sinkhorn,
        h_aggregate_fwd,
        h_aggregate_bwd,
        h_post_bda_fwd,
        h_post_bda_bwd,
        proj_rms_compute_h,
    )
    message = (
        f"MHC_FORCE_BACKEND={_MHC_FORCED_BACKEND}; "
        f"sinkhorn={sinkhorn}; "
        f"h_aggregate=fwd:{h_aggregate_fwd},bwd:{h_aggregate_bwd}; "
        f"h_post_bda=fwd:{h_post_bda_fwd},bwd:{h_post_bda_bwd}; "
        f"proj_rms_compute_h={proj_rms_compute_h}"
    )
    return message, all(backend == "native" for backend in selected)


def fused_sinkhorn(input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
    """Project logits to a doubly stochastic matrix using Hip, Triton, cuTile, then torch."""
    _raise_mhc_backend_validation_error()
    if not input_logits.is_cuda or input_logits.numel() == 0:
        return native_sinkhorn(input_logits, num_iterations, eps)
    triton_sinkhorn = _get_triton_sinkhorn()
    if is_hip_available():
        return HipFusedSinkhorn.apply(input_logits, num_iterations, eps)
    if triton_sinkhorn is not None:
        return triton_sinkhorn(input_logits, num_iterations, eps)
    if is_cutile_available():
        return CutileSinkhornKnopp.apply(input_logits, num_iterations, eps)
    return native_sinkhorn(input_logits, num_iterations, eps)


def fused_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    """Weighted n-stream to 1-stream aggregation using Hip/Triton/cuTile/torch."""
    _raise_mhc_backend_validation_error()
    if x.is_cuda and x.numel() and is_hip_available():
        return HipFusedHAggregate.apply(x, h_pre)
    if x.is_cuda and x.numel() and (_TRITON_AVAILABLE or is_cutile_available()):
        return FusedHAggregate.apply(x, h_pre)
    return native_h_aggregate(x, h_pre)


def fused_h_aggregate_into(x: Tensor, h_pre: Tensor, out: Tensor) -> Tensor:
    """Weighted aggregation that writes directly to fixed-address ``out``."""
    _raise_mhc_backend_validation_error()
    if x.is_cuda and x.numel() and is_hip_available():
        return HipFusedHAggregateInto.apply(x, h_pre, out)
    if x.is_cuda and x.numel() and (_TRITON_AVAILABLE or is_cutile_available()):
        return FusedHAggregateInto.apply(x, h_pre, out)
    from megatron.core.transformer.hyper_connection import native_h_aggregate_into

    return native_h_aggregate_into(x, h_pre, out)


def fused_h_post_bda(
    h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    """Fused H_res.T @ residual + H_post * (x + bias)."""
    _raise_mhc_backend_validation_error()
    if x.is_cuda and x.numel() and is_hip_available():
        return HipFusedHPostBDA.apply(h_res, original_residual, h_post, x, bias)
    if x.is_cuda and x.numel() and (_TRITON_AVAILABLE or is_cutile_available()):
        return FusedHPostBDA.apply(h_res, original_residual, h_post, x, bias)
    return native_h_post_bda(h_res, original_residual, h_post, x, bias)


def fused_proj_rms_compute_h(
    x: Tensor,
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    eps: float = 1e-6,
    compute_h_eps: float = 1e-6,
    *,
    eps_inside_sqrt: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Projection + RMS norm + compute_h split outputs using hip, cuTile, then torch.

    ``eps_inside_sqrt`` selects single-pass mHC normalization. In that mode the
    fourth output is ``sqrt(mean(x**2) + eps)``; otherwise it is the unregularized
    RMS and epsilon is added after the square root, preserving legacy mHC.
    """
    _raise_mhc_backend_validation_error()
    if x.is_cuda and x.numel() and is_hip_available():
        return HipFusedProjRmsComputeH.apply(
            x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps, compute_h_eps
        )
    if x.is_cuda and x.numel() and is_cutile_available():
        return CutileProjRmsComputeH.apply(
            x,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            n,
            eps,
            compute_h_eps,
            eps_inside_sqrt,
        )
    return _torch_proj_rms_compute_h(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps, compute_h_eps, eps_inside_sqrt
    )