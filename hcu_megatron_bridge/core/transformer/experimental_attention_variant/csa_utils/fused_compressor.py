# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from typing import Optional

_MINIMUM_COMPUTE_CAPABILITY_MAJOR = 9
_UNINITIALIZED = object()
# Lazily resolved ``cudnn.csa.compressor`` module: ``_UNINITIALIZED`` -> not probed yet,
# ``None`` -> probed and unavailable (reason kept in ``_frontend_error``).
_frontend = _UNINITIALIZED
_frontend_error: Optional[Exception] = None

def _get_frontend():
    """Return cudnn-frontend's ``cudnn.csa.compressor`` module, or None if unavailable.

    Probed lazily (on the first dispatch call, not at Megatron import) by importing the
    concrete entry points rather than comparing version numbers: any install that has
    them can serve the dispatch, and ``nvidia-cudnn-frontend`` releases that predate the
    CSA compressor API (cudnn-frontend #427) lack ``cudnn.csa`` and fall back to eager
    naturally. Any import-time failure — missing package, missing ``nvidia-cutlass-dsl``
    (the frontend's ``cutedsl`` extra), a partially installed or ABI-incompatible
    frontend — also degrades to eager instead of breaking ``csa.py``'s import chain; the
    error is kept in ``_frontend_error`` for debugging.
    """
    global _frontend, _frontend_error
    if _frontend is _UNINITIALIZED:
        try:
            from cudnn.csa import compressor as fe_compressor

            fe_compressor.csa_compressor_forward_wrapper
            fe_compressor.csa_compressor_backward_wrapper
            _frontend = fe_compressor
        except Exception as cudnn_error:
            try:
                import hip_dsa_kernels as _hk

                _hk.csa_compressor_forward_wrapper
                _hk.csa_compressor_backward_wrapper
                _frontend = _hk
            except Exception as hip_error:
                _frontend = None
                _frontend_error = (cudnn_error, hip_error)
    return _frontend