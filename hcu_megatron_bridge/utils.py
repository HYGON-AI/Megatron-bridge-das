# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import os
import sys
import functools
from typing import Any, Callable, Optional, Union

import torch


def print_only_rank0(msg: str):
    """
    Print a message only if the current process is rank 0.

    Args:
        msg (str): The message to print.
    """
    rank = int(
        os.environ.get("RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("OMPI_COMM_WORLD_RANK")
        or "0"
    )
    if rank == 0:
        print(msg, flush=True)


def _raw_attr(path: str) -> Any:
    """
    Return the attribute stored on the owner of ``path`` without triggering the
    descriptor protocol, so ``staticmethod`` / ``classmethod`` stay visible.

    ``path`` mixes a module path with optional class names, e.g.
    ``pkg.mod.Cls.method``, so the longest already-imported module prefix is
    used to locate the owner.
    """
    parts = path.split('.')
    for i in range(len(parts) - 1, 0, -1):
        module = sys.modules.get('.'.join(parts[:i]))
        if module is None:
            continue
        owner = functools.reduce(getattr, parts[i:-1], module)
        return vars(owner).get(parts[-1])
    return None


def profiler_annotate_patch(path: str, message: str) -> Callable:
    """
    Build an ``apply_wrapper`` callable that annotates ``path`` for the profiler
    while preserving whether the original attribute was a plain function, a
    ``staticmethod``, or a ``classmethod``.

    ``torch.compile`` is left in place so the profile reflects compiled
    performance.  A consequence is that the range only covers the CPU-side
    launch of a compiled region, and never covers its backward.
    """

    def wrapper(func):
        annotated = profiler_annotate(message)(func)
        raw = _raw_attr(path)
        if isinstance(raw, (staticmethod, classmethod)):
            return type(raw)(annotated)
        return annotated

    return wrapper


def profiler_annotate(message: Optional[Union[str, Callable]] = None) -> Callable:
    target_func = message if callable(message) else None
    profile_message = None if target_func is not None else message

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            from torch.autograd import profiler

            if not getattr(profiler, "_is_profiler_enabled", False):
                return func(*func_args, **func_kwargs)
            
            profile_name = profile_message if profile_message is not None else func.__name__

            with torch.profiler.record_function(profile_name):
                return func(*func_args, **func_kwargs)

        return wrapper

    if target_func is not None:
        return decorator(target_func)

    return decorator


# ============================ model dump (debug) ============================
# Temporary debug helper: print the constructed model structure at startup.
# Used by run_recipe.py (main() -> _register_model_dump_hook(recipe)).
# Delete this block and that call site to restore the original behavior.
def _dump_model_structure(model, *, name="model", max_lines=400):
    """Print a Megatron model structure. ``model`` may be one module or a PP stage list."""
    wrappers = []
    try:
        from megatron.core.transformer.module import Float16Module

        wrappers.append(Float16Module)
    except Exception:
        pass
    for module_name, class_name in (
        ("megatron.core.distributed", "DistributedDataParallel"),
        ("torch.nn.parallel", "DistributedDataParallel"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            wrappers.append(getattr(module, class_name))
        except Exception:
            pass

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for index, chunk in enumerate(chunks):
        module = chunk
        while wrappers and isinstance(module, tuple(wrappers)):
            module = module.module
        text = repr(module)
        lines = text.splitlines()
        if len(lines) > max_lines:
            body = "\n".join(lines[:max_lines]) + "\n... <truncated %d more lines>" % (
                len(lines) - max_lines
            )
        else:
            body = text
        print_only_rank0("\n===== %s[%d] : %s =====\n%s" % (name, index, type(module).__name__, body))
    return model


def _register_model_dump_hook(cfg):
    """Attach _dump_model_structure to the model config's pre-wrap hooks."""
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        print_only_rank0("[model-dump] WARNING: cfg.model is missing, skip model dump")
        return
    hooks = getattr(model_cfg, "pre_wrap_hooks", None)
    if hooks is not None:
        if _dump_model_structure not in hooks:
            hooks.append(_dump_model_structure)
        print_only_rank0("[model-dump] registered pre-wrap hook on %s" % type(model_cfg).__name__)
        return
    register = getattr(model_cfg, "register_pre_wrap_hook", None)
    if callable(register):
        register(_dump_model_structure)
        print_only_rank0("[model-dump] registered pre-wrap hook via register_pre_wrap_hook")
        return
    print_only_rank0("[model-dump] WARNING: model config supports no pre-wrap hook, skip model dump")
# ========================== end model dump (debug) ==========================
