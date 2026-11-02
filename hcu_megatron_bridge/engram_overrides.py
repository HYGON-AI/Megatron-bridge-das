# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Apply DeepSeek-V4.1 Engram CLI overrides that the config schema does not declare.

``process_config_with_overrides()`` turns the ConfigContainer dataclass into a **struct**
OmegaConf config, so ``model.engram_layer_ids=[2,4]`` raises::

    ConfigAttributeError: Key 'engram_layer_ids' is not in struct

Rather than adding fields to Megatron-Bridge's dataclasses (a 3rdparty edit), this shim
pulls the Engram keys out of the override list, lets the original implementation handle
everything else, and finally ``setattr``s the values onto the provider object -- which is
exactly what the bridge's ``_provider_override()`` reads at model-build time.

Pure stdlib so it can be unit-tested without torch/megatron.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Iterable

# Keys consumed by hcu_megatron_bridge's DeepSeek-V4.1 bridge.
ENGRAM_KEYS = ("engram_layer_ids", "engram_vocab_sizes")


def parse_override_value(text: str) -> Any:
    """Parse ``[2,4]`` / ``(2,4)`` / ``4096`` / ``a,b`` into a Python value."""
    text = text.strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # Hydra allows bare comma lists such as model.engram_layer_ids=2,4
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) > 1:
            try:
                return [int(part) for part in parts]
            except ValueError:
                return parts
        return text


def split_overrides(cli_overrides: Iterable[str]) -> tuple[list[str], dict[str, Any]]:
    """Split ``model.<engram key>=...`` overrides out of the Hydra override list."""
    kept: list[str] = []
    extra: dict[str, Any] = {}
    for override in cli_overrides:
        raw = override.lstrip("+")
        for key in ENGRAM_KEYS:
            prefix = f"model.{key}="
            if raw.startswith(prefix):
                extra[key] = parse_override_value(raw[len(prefix):])
                break
        else:
            kept.append(override)
    return kept, extra


def wrap_process_config(original: Callable) -> Callable:
    """Wrap ``process_config_with_overrides`` so Engram keys bypass the struct check.

    NOTE: ``MegatronPatchesManager.register_patch`` classifies a callable as a *decorator*
    when its ``__name__`` ends with ``wrapper``/``decorator`` and then applies it as
    ``fn(original)``.  The returned function is therefore named explicitly, otherwise the
    manager would call it with the original function object instead of a config.
    """

    def process_config_with_overrides_hcu(config, config_filepath=None, cli_overrides=None):
        kept, extra = split_overrides(cli_overrides or [])
        result = original(
            config, config_filepath=config_filepath, cli_overrides=kept or None
        )
        model_cfg = getattr(result, "model", None)
        for key, value in extra.items():
            if model_cfg is not None and value is not None:
                setattr(model_cfg, key, value)
        return result

    return process_config_with_overrides_hcu
