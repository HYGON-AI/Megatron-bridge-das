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

"""Bridge for the DeepSeek-V4.1 model family (DeepSeek-V4 + Engram).

DeepSeek-V4.1 keeps the whole V4 layout -- hybrid MLA / CSA attention,
hyper-connections (mHC), hash-routed MoE, MTP -- and adds **Engram**, the
N-gram conditional-memory module (``megatron/core/models/engram``).  This
bridge therefore subclasses :class:`DeepSeekV4Bridge` and only contributes the
Engram-specific wiring:

1. derive an :class:`EngramConfig` from the HF config,
2. hand every decoder layer spec an Engram composition point,
3. map the Engram weights of the HF checkpoint.

HF checkpoint layout (per Engram layer)::

    layers.N.engram.embed.weight          [sum(table_sizes), memory_dim]  (FP8_E4M3)
    layers.N.engram.embed.weight_scale_inv[ceil(rows/128), ceil(dim/128)]
    layers.N.engram.wkv.weight            [num_streams*hidden + hidden, memory_dim]
    layers.N.engram.wkv.weight_scale_inv
    layers.N.engram.q_weight              [num_streams, memory_dim]
    layers.N.engram.k_weight              [num_streams, memory_dim]

Megatron side (Engram module)::

    engram.embedding.tables.{order}.weight   EP-sharded rows, one table per n-gram order
    engram.key_projection.weight             [num_streams*hidden, memory_dim]
    engram.value_projection.weight           [hidden, memory_dim]
    engram.query_norm.weight / key_norm.weight

Runtime prerequisites (supplied by the pinned Megatron-Core):

* ``megatron.core.models.engram`` (EngramConfig / Engram / layer_specs)
* ``TransformerLayerSubmodules.engram`` + ``_maybe_apply_engram``
* ``--engram-*`` arguments

Known limitations of this first milestone (see the module TODO list at the end):
pipeline-parallel token prefetch for Engram layers and full FP8 table import are
not wired yet.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge
from megatron.bridge.models.mla_provider import MLAModelProvider

logger = logging.getLogger(__name__)



try:  # pragma: no cover - depends on the pinned Megatron-Core
    from megatron.core.models.engram import EngramConfig, apply_engram_to_layer_spec

    HAVE_ENGRAM = True
except ImportError:  # pragma: no cover
    EngramConfig = None
    apply_engram_to_layer_spec = None
    HAVE_ENGRAM = False


# Hyper-parameters that the DeepSeek-V4.1 reference implementation fixes and that
# the HF config does not carry explicitly.
_ENGRAM_DEFAULTS = {
    "max_ngram_order": 4,
    "num_hash_heads": 8,
    "memory_dim": 256,
    "kernel_size": 4,
    "hash_seed": 0,
    "variant": "deepseek",
    "embedding_lr_multiplier": 5.0,
    "embedding_weight_decay": 0.0,
}


def has_engram() -> bool:
    """Return whether the installed Megatron-Core provides the Engram extension."""
    return HAVE_ENGRAM


def _get(hf_config: Any, *names: str, default: Any = None) -> Any:
    """Return the first present attribute among ``names`` (native vs legacy HF names)."""
    for name in names:
        value = getattr(hf_config, name, None)
        if value is not None:
            return value
    return default


def _provider_override(provider: Any, name: str, default: Any = None) -> Any:
    """Return a provider-level Engram override when the recipe/CLI set one."""
    if provider is None:
        return default
    value = getattr(provider, name, None)
    return default if value is None else value


def engram_settings_from_hf(
    hf_config: Any, model_dir: str | None = None, provider: Any = None
) -> Optional[Dict[str, Any]]:
    """Extract Engram hyper-parameters, preferring provider-level overrides.

    Field semantics follow the reference V4.1 adapter:

    * ``engram_vocab_size`` is the *per-order* table row budget, repeated for every
      n-gram order ``2..engram_max_ngram_size``;
    * ``engram_num_embeddings`` is the checkpoint's per-layer row total and is only
      used to back-derive the per-order size when ``engram_vocab_size`` is absent;
    * HF layer IDs are 0-based while Megatron layer numbers are 1-based, so the HF
      list is shifted by one.  A provider override (``engram_layer_ids``) is already
      in Megatron numbering and is used verbatim -- that is what lets a small test
      recipe shrink the backbone without regenerating the checkpoint config.

    ``provider`` is the (possibly still being built) TransformerConfig, so this may
    be called again at model-build time to pick up recipe/CLI overrides.
    """
    hf_layer_ids = _get(hf_config, "engram_layer_ids")
    override_layers = _provider_override(provider, "engram_layer_ids")
    if not hf_layer_ids and override_layers is None:
        return None

    max_ngram_order = int(
        _provider_override(
            provider,
            "engram_max_ngram_order",
            _get(
                hf_config,
                "engram_max_ngram_size",
                "engram_max_ngram_order",
                default=_ENGRAM_DEFAULTS["max_ngram_order"],
            ),
        )
    )
    if max_ngram_order < 2:
        raise ValueError(f"engram_max_ngram_size must be >= 2, got {max_ngram_order}.")

    num_hash_heads = int(
        _provider_override(
            provider,
            "engram_n_heads",
            _get(hf_config, "engram_n_heads", default=_ENGRAM_DEFAULTS["num_hash_heads"]),
        )
    )
    head_dim = int(
        _provider_override(
            provider,
            "engram_head_dim",
            _get(hf_config, "engram_head_dim", default=_ENGRAM_DEFAULTS["memory_dim"]),
        )
    )

    # ``global_vocab_sizes`` is the *per-order* row budget for n-gram orders 2..max,
    # i.e. exactly what dsv41's ``--engram-vocab-sizes`` takes.  The checkpoint's
    # ``engram_num_embeddings`` is a different quantity: the per-layer row total, which
    # equals sum(vocab_sizes) * num_hash_heads (V4.1 Flash: 3 * 16M * 8 = 384M).  It is
    # therefore only used to back-derive the base when the scalar is missing.
    per_order_sizes = _provider_override(provider, "engram_vocab_sizes")
    if isinstance(per_order_sizes, (list, tuple)) and len(per_order_sizes) == max_ngram_order - 1:
        global_vocab_sizes = tuple(int(size) for size in per_order_sizes)
    else:
        if per_order_sizes is not None:
            logger.warning(
                "engram_vocab_sizes needs %d entries for n-gram orders 2..%d, got %s; "
                "falling back to engram_vocab_size.",
                max_ngram_order - 1,
                max_ngram_order,
                list(per_order_sizes),
            )
        vocab_size = _provider_override(
            provider, "engram_vocab_size", _get(hf_config, "engram_vocab_size")
        )
        if vocab_size is None:
            rows = _get(hf_config, "engram_num_embeddings") or []
            if not rows:
                raise ValueError(
                    "DeepSeek-V4.1 Engram config needs 'engram_vocab_size' "
                    "(or a non-empty 'engram_num_embeddings' to derive it)."
                )
            divisor = max(1, (max_ngram_order - 1) * num_hash_heads)
            vocab_size = int(max(rows)) // divisor
            logger.warning(
                "DeepSeek-V4.1 config has no 'engram_vocab_size'; derived %d rows per "
                "order/head from engram_num_embeddings=%s (divisor %d).",
                vocab_size,
                list(rows),
                divisor,
            )
        global_vocab_sizes = tuple(int(vocab_size) for _ in range(max_ngram_order - 1))

    # Resolution order: provider override (in-process recipe) -> env var (CI / test
    # artefacts, avoids adding a field to the OmegaConf struct schema) -> HF config ->
    # conventional <model_dir>/engram_tokenizer_map.json.
    tokenizer_map = str(
        _provider_override(
            provider,
            "engram_tokenizer_map",
            os.environ.get("V41_ENGRAM_TOKENIZER_MAP", "")
            or _get(hf_config, "engram_tokenizer_map", default="")
            or "",
        )
        or ""
    )
    if not tokenizer_map and model_dir:
        candidate = os.path.join(str(model_dir), "engram_tokenizer_map.json")
        if os.path.isfile(candidate):
            tokenizer_map = candidate
    if not tokenizer_map:
        logger.warning(
            "DeepSeek-V4.1 Engram config has no 'engram_tokenizer_map'; the Engram "
            "variant that needs the offline tokenizer artifact will fail validation."
        )

    layer_ids = (
        tuple(int(layer_id) for layer_id in override_layers)
        if override_layers is not None
        else tuple(int(layer_id) + 1 for layer_id in hf_layer_ids)
    )

    return {
        "global_vocab_sizes": global_vocab_sizes,
        "layer_ids": layer_ids,
        "max_ngram_order": max_ngram_order,
        "num_hash_heads": num_hash_heads,
        "memory_dim": num_hash_heads * head_dim,
        "kernel_size": int(
            _provider_override(
                provider,
                "engram_kernel_size",
                _get(hf_config, "engram_kernel_size", default=_ENGRAM_DEFAULTS["kernel_size"]),
            )
        ),
        "hash_seed": int(_ENGRAM_DEFAULTS["hash_seed"]),
        "boundary_token_id": int(
            _provider_override(
                provider,
                "engram_pad_token_id",
                _get(hf_config, "engram_pad_token_id", "pad_token_id", default=0),
            )
        ),
        "tokenizer_map_path": tokenizer_map,
        "variant": str(_get(hf_config, "engram_variant", default=_ENGRAM_DEFAULTS["variant"])),
        "embedding_lr_multiplier": float(_ENGRAM_DEFAULTS["embedding_lr_multiplier"]),
        "embedding_weight_decay": float(_ENGRAM_DEFAULTS["embedding_weight_decay"]),
    }


class _EngramParallelismView:
    """Expose a *resolved* parallelism view to Engram's startup validators.

    ``expert_tensor_parallel_size`` is optional in Megatron and means "same as TP"
    when unset, but Engram compares it against ``tensor_model_parallel_size``
    directly.  Resolving it here keeps the check meaningful without mutating the
    provider (CLI overrides are applied after the bridge runs).
    """

    def __init__(self, config: Any):
        self._config = config

    @property
    def expert_tensor_parallel_size(self):
        value = getattr(self._config, "expert_tensor_parallel_size", None)
        if value is None:
            value = getattr(self._config, "tensor_model_parallel_size", 1)
        return value

    def __getattr__(self, name):
        return getattr(self._config, name)


def _validate_engram_startup(engram_config: Any, transformer_config: Any) -> None:
    """Run Engram's startup checks against the *final* TransformerConfig."""
    engram_config.validate_startup(
        _EngramParallelismView(transformer_config),
        expected_tokenizer_vocab_size=(
            getattr(transformer_config, "actual_vocab_size", None)
            or getattr(transformer_config, "padded_vocab_size", None)
        ),
        packed_sequences=bool(
            getattr(transformer_config, "enable_offline_packing", False)
        ),
        use_fsdp=bool(
            getattr(transformer_config, "use_megatron_fsdp", False)
            or getattr(transformer_config, "use_torch_fsdp2", False)
        ),
        sequence_length=getattr(transformer_config, "seq_length", None),
    )


def build_engram_config(
    hf_config: Any, transformer_config: Any, model_dir: str | None = None
) -> Optional[Any]:
    """Build an :class:`EngramConfig` from ``transformer_config`` (``None`` if no Engram)."""
    if not HAVE_ENGRAM:
        raise RuntimeError(
            "DeepSeek-V4.1 Engram requires Megatron-Core with the Engram extension "
            "(megatron.core.models.engram). Forward the Megatron-LM dsv41 tree into "
            "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM first."
        )

    settings = engram_settings_from_hf(
        hf_config, model_dir=model_dir, provider=transformer_config
    )
    if settings is None:
        return None

    engram_config = EngramConfig(**settings)
    # Parallelism/feature checks run at model-build time (apply_engram_to_provider):
    # the CLI overrides that set TP/ETP/bf16/VPP are applied after the bridge runs.
    return engram_config


def apply_engram_to_provider(
    provider: MLAModelProvider, hf_config: Any, model_dir: str | None = None
) -> None:
    """Attach Engram to every decoder layer spec produced by ``provider``.

    ``provider.transformer_layer_spec`` is a *callable* (the experimental
    attention-variant block spec builder), so the Engram composition point is added by
    wrapping that builder.  The Engram configuration is built **inside** the wrapper:
    the recipe/CLI overrides (layer placement, table sizes, tokenizer map) are applied
    only after ``provider_bridge`` returns, so an eagerly built config would ignore
    them and fail placement validation.
    """
    if engram_settings_from_hf(hf_config, model_dir=model_dir, provider=provider) is None:
        logger.info("DeepSeek-V4.1 checkpoint has no Engram layers; building a plain V4 model.")
        return

    base_builder = provider.transformer_layer_spec
    if not callable(base_builder):
        raise TypeError(
            "DeepSeek-V4.1 Engram expects provider.transformer_layer_spec to be a "
            f"spec builder callable, got {type(base_builder).__name__}."
        )

    state = {"engram_config": None, "validated": False}

    def engram_layer_spec_builder(config):
        engram_config = state["engram_config"]
        if engram_config is None:
            engram_config = build_engram_config(hf_config, config, model_dir=model_dir)
            if engram_config is None:
                return base_builder(config)
            state["engram_config"] = engram_config
            provider.engram_config = engram_config
            logger.info(
                "Engram enabled: layers=%s orders=%s tables=%s memory_dim=%d heads=%d",
                list(engram_config.layer_ids),
                engram_config.max_ngram_order,
                list(engram_config.global_vocab_sizes),
                engram_config.memory_dim,
                engram_config.num_hash_heads,
            )
            table_bytes = (
                sum(engram_config.global_vocab_sizes)
                * engram_config.memory_dim
                * 2  # bf16 table storage
            )
            logger.info(
                "Engram table footprint: %.2f GiB per layer before EP sharding (EP=%d)",
                table_bytes / 1024**3,
                int(getattr(config, "expert_model_parallel_size", 1) or 1),
            )
        if not state["validated"]:
            _validate_engram_startup(engram_config, config)
            state["validated"] = True
        return apply_engram_to_layer_spec(base_builder(config), engram_config)

    provider.transformer_layer_spec = engram_layer_spec_builder
    provider.engram_enabled = True


@MegatronModelBridge.register_bridge(
    source="DeepseekV41ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v41",
)
class DeepSeekV41Bridge(DeepSeekV4Bridge):
    """Megatron Bridge implementation for DeepSeek-V4.1 (V4 + Engram)."""

    # ------------------------------------------------------------------
    # Provider configuration
    # ------------------------------------------------------------------

    def provider_bridge(self, hf_pretrained) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # ---- V4.1: CSA2 source-layer routing + hyper-connection epsilon ----
        # TransformerConfig defaults to dsv4_version="v4"; V4.1 checkpoints route KV /
        # index / candidate blocks from earlier "source" layers, and its config.json
        # carries those layer lists (0-based, matching CSA2's layer_idx = layer_number-1).
        provider.dsv4_version = "v4.1"
        provider.csa2_kv_source_layers = list(
            _get(hf_config, "kv_source_layer_ids") or []
        )
        provider.csa2_index_source_layers = list(
            _get(hf_config, "index_source_layer_ids") or []
        )
        provider.csa2_candidate_source_layer = _get(
            hf_config, "candidate_source_layer_id"
        )
        provider.csa2_candidate_topk_blocks = int(
            _get(hf_config, "candidate_topk_blocks", default=0) or 0
        )
        provider.csa2_candidate_block_size = int(
            _get(hf_config, "candidate_block_size", default=0) or 0
        )
        provider.mhc_epsilon = float(_get(hf_config, "hc_eps", default=1e-6))

        # ---- V4.1 backbone constraints (belongs here, not only in the recipe) ----
        # V4.1 forbids MTP/DSpark layers, but the V4 bridge copies
        # num_nextn_predict_layers into mtp_num_layers.
        provider.mtp_num_layers = None
        # The V4.1 validator requires exactly num_layers ratios, while V4 slices
        # num_hidden_layers + num_nextn_predict_layers (40 + 3 = 43 here).
        ratios = list(provider.csa_compress_ratios or [])
        if ratios:
            provider.csa_compress_ratios = ratios[: provider.num_layers]
        # Hash routing is a V4-Flash feature; V4.1 routes every layer through MoE.
        provider.moe_n_hash_layers = 0

        model_dir = getattr(hf_pretrained, "model_name_or_path", None) or getattr(
            hf_config, "name_or_path", None
        )
        apply_engram_to_provider(provider, hf_config, model_dir=model_dir)
        return provider

    # ------------------------------------------------------------------
    # Export: HF config reconstruction
    # ------------------------------------------------------------------

    @classmethod
    def megatron_to_hf_config(cls, provider: MLAModelProvider) -> dict:
        hf_cfg = super().megatron_to_hf_config(provider)
        engram_config = getattr(provider, "engram_config", None)
        if engram_config is None:
            return hf_cfg

        # Convert back to the official HF conventions: 0-based layer ids, per-head
        # head_dim, and per-layer total row counts (sum over orders and hash heads).
        hf_cfg["engram_layer_ids"] = [int(layer_id) - 1 for layer_id in engram_config.layer_ids]
        hf_cfg["engram_num_embeddings"] = [
            int(sum(engram_config.table_sizes(layer_id)))
            for layer_id in engram_config.layer_ids
        ]
        hf_cfg["engram_max_ngram_size"] = engram_config.max_ngram_order
        hf_cfg["engram_n_heads"] = engram_config.num_hash_heads
        hf_cfg["engram_head_dim"] = engram_config.memory_dim // max(
            1, engram_config.num_hash_heads
        )
        hf_cfg["engram_pad_token_id"] = engram_config.boundary_token_id
        hf_cfg["engram_tokenizer_map"] = engram_config.tokenizer_map_path
        return hf_cfg

    # ------------------------------------------------------------------
    # Weight mapping
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:  # noqa: C901
        mappings = list(super().mapping_registry().mappings)

        # Engram dense projections.  ``wkv`` packs key and value rows in one HF
        # tensor; mapping it directly is wrong, so both halves are consumed by
        # the dedicated importer (see TODO below) and are intentionally absent
        # from the generic registry.
        mappings += [
            # The per-order tables are EP-row-sharded and FP8-quantised; they are
            # imported by the dedicated Engram loader instead of a generic mapping.
            AutoMapping(
                "decoder.layers.*.engram.conv_norm.weight",
                "layers.*.engram.conv_norm.weight",
            )
        ]
        return MegatronMappingRegistry(*mappings)

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param,
        hf_state_dict: Mapping[str, torch.Tensor],
    ):
        """Dequantise V4.1 tensors with the checkpoint's own FP8 block geometry.

        V4 stores attention/embedding FP8 weights with 128x128 block scales; V4.1 uses
        32x32 blocks (``quantization_config.weight_block_size``), so the upstream
        ``FP8_BLOCK_SIZE = 128`` default would mis-shape every scale tensor.
        ``engram_weights.dequantize_fp8`` infers the geometry from the shapes.
        """
        from .engram_weights import dequantize_fp8

        if isinstance(hf_param, dict):
            return {
                key: self.maybe_modify_loaded_hf_weight(value, hf_state_dict)
                for key, value in hf_param.items()
            }
        if not isinstance(hf_param, str) or hf_param not in hf_state_dict:
            return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

        weight = hf_state_dict[hf_param]
        if weight.dtype != torch.float8_e4m3fn:
            # BF16 tensors and packed MXFP4 experts keep the upstream handling.
            return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

        scale_key = hf_param[: -len(".weight")] + ".scale"
        scale = hf_state_dict.get(scale_key)
        if scale is None:
            return weight.to(torch.bfloat16)
        return dequantize_fp8(weight, scale).to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Engram weight import
    # ------------------------------------------------------------------
    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model,
        allowed_mismatched_params=None,
    ):
        """Load HF weights, then stream the Engram tensors that the generic loader skips.

        The six Engram tensors per layer are deliberately absent from ``mapping_registry``:
        ``engram.embed.weight`` is ~98 GiB inside a single 95 GiB shard, so it has to be
        read with ``safe_open(...).get_slice()`` row windows and scattered into the EP-local
        prime tables instead of being materialised by the shard loader.
        """
        models = super().load_weights_hf_to_megatron(
            hf_pretrained, megatron_model, allowed_mismatched_params
        )
        self._load_engram_weights(models, hf_pretrained)
        return models

    def _load_engram_weights(self, megatron_model, hf_pretrained) -> None:
        from .engram_weights import (
            has_engram_tensors,
            iter_table_chunks,
            read_tensor,
            resolve_weight_map,
            split_wkv,
            table_row_ranges,
            tensor_shape,
        )

        model_dir = getattr(hf_pretrained, "model_name_or_path", None) or getattr(
            self.hf_config, "name_or_path", None
        )
        if not model_dir:
            logger.warning("Engram import skipped: checkpoint directory is unknown.")
            return

        root, weight_map = resolve_weight_map(model_dir)
        if not has_engram_tensors(weight_map):
            return

        from megatron.core.utils import unwrap_model

        chunks = unwrap_model(megatron_model)
        if not isinstance(chunks, (list, tuple)):
            chunks = [chunks]

        loaded_layers = 0
        for chunk in chunks:
            for _, module in chunk.named_modules():
                if not hasattr(module, "engram_config") or not hasattr(module, "embedding"):
                    continue
                layer_number = getattr(module, "layer_number", None)
                if layer_number is None:
                    continue
                hf_layer = int(layer_number) - 1
                prefix = f"layers.{hf_layer}.engram."
                if prefix + "embed.weight" not in weight_map:
                    continue

                tables = list(module.embedding.tables)
                sizes = [int(size) for size in module.table_sizes.tolist()]
                # Guard: the checkpoint must hold exactly the rows the prime tables allocate,
                # otherwise the tokenizer map / vocab sizes disagree and every table offset
                # downstream would be silently wrong.
                checkpoint_rows = tensor_shape(root, weight_map, prefix + "embed.weight")[0]
                if checkpoint_rows != sum(sizes):
                    raise ValueError(
                        f"Engram layer {layer_number}: checkpoint has {checkpoint_rows} table "
                        f"rows but the prime allocation needs {sum(sizes)}. Check "
                        f"engram_vocab_sizes / engram_vocab_size against config.json."
                    )
                ranges = table_row_ranges(sizes, module.embedding.ep_size, module.embedding.ep_rank)
                if len(ranges) != len(tables):
                    raise ValueError(
                        f"Engram layer {layer_number}: {len(tables)} EP tables but "
                        f"{len(ranges)} row ranges"
                    )

                for table, (hf_start, local_rows) in zip(tables, ranges):
                    if local_rows != table.weight.shape[0]:
                        raise ValueError(
                            f"Engram layer {layer_number}: table {table.weight.shape} does not "
                            f"match the EP shard ({local_rows} rows)"
                        )
                    destination = table.weight.data
                    for offset, rows in iter_table_chunks(
                        root,
                        weight_map,
                        prefix + "embed.weight",
                        prefix + "embed.scale" if prefix + "embed.scale" in weight_map else None,
                        hf_start,
                        local_rows,
                    ):
                        destination[offset : offset + rows.shape[0]].copy_(
                            rows.to(device=destination.device, dtype=destination.dtype)
                        )

                wkv_key = prefix + "wkv.weight"
                if wkv_key in weight_map:
                    from .engram_weights import dequantize_fp8

                    scale_key = prefix + "wkv.scale"
                    wkv = read_tensor(root, weight_map, wkv_key)
                    scale = (
                        read_tensor(root, weight_map, scale_key)
                        if scale_key in weight_map
                        else None
                    )
                    wkv = dequantize_fp8(wkv, scale)
                    key_rows = int(module.num_streams * module.hidden_size)
                    key, value = split_wkv(wkv, key_rows)
                    module.key_projection.weight.data.copy_(
                        key.to(
                            device=module.key_projection.weight.device,
                            dtype=module.key_projection.weight.dtype,
                        )
                    )
                    module.value_projection.weight.data.copy_(
                        value.to(
                            device=module.value_projection.weight.device,
                            dtype=module.value_projection.weight.dtype,
                        )
                    )
                    del wkv, key, value

                for hf_suffix, parameter in (
                    ("q_weight", module.query_norm.weight),
                    ("k_weight", module.key_norm.weight),
                ):
                    key = prefix + hf_suffix
                    if key in weight_map:
                        tensor = read_tensor(root, weight_map, key).reshape(-1)
                        parameter.data.copy_(
                            tensor.to(device=parameter.device, dtype=parameter.dtype)
                        )

                loaded_layers += 1

        if loaded_layers:
            logger.info(
                "Imported Engram weights for %d layer(s) from %s", loaded_layers, root
            )
            logger.warning(
                "Engram short_conv/conv_norm have no checkpoint tensor; they keep their "
                "Megatron initialisation."
            )

    # ------------------------------------------------------------------
    # Remaining Engram gaps
    # ------------------------------------------------------------------
    # * ``short_conv`` / ``conv_norm`` have no checkpoint tensors (the official
    #   implementation has no short convolution), so they keep their Megatron
    #   initialisation.  If the reference model turns out to use them, this needs a
    #   follow-up.
    # * Pipeline-parallel Engram token prefetch in the training step is still missing
    #   (PP is disabled for V4.1 anyway: TransformerBlock rejects PP>1).
    # * ``engram.embedding`` expects the EP-local rows; the importer streams them with
    #   ``safe_open(...).get_slice()`` so a 95 GiB shard is never materialised.


# ---------------------------------------------------------------------------
# DeepSeek-V4.1 HF config registration
# ---------------------------------------------------------------------------
# ``AutoConfig.from_pretrained()`` resolves ``config.json``'s ``model_type`` through
# ``CONFIG_MAPPING``.  DeepSeek-V4.1 uses ``deepseek_v41`` (composite: text + vision)
# plus its sub-configs ``deepseek_v41_text`` / ``deepseek_v41_vision``, none of which
# older transformers releases know about, so loading a V4.1 checkpoint fails with::
#
#     KeyError: 'deepseek_v41'
#     ValueError: ... model type `deepseek_v41` but Transformers does not recognize ...
#
# The registration mirrors the ms-swift loader: the text config subclasses the native
# ``DeepseekV4Config`` (so V4's ``__post_init__``, layer-type validation and derived
# fields keep working), widens the CSA2 compress-ratio vocabulary, and keeps the raw
# ``compress_ratios`` for the Megatron-Core side.

_HF_COMPAT_INSTALLED = False

# CSA2 widens the per-layer ratio vocabulary beyond V4's {0, 4, 128}.
_CSA2_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    1: "compressed_sparse_attention",
    2: "heavily_compressed_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


def _install_hf_config_compat() -> None:
    """Idempotently register the DeepSeek-V4.1 config classes."""
    global _HF_COMPAT_INSTALLED
    if _HF_COMPAT_INSTALLED:
        return
    _HF_COMPAT_INSTALLED = True

    try:
        from transformers import AutoConfig, PretrainedConfig
    except Exception:  # transformers not installed (pure Megatron environment)
        return

    try:  # V4.1 text config should inherit V4's derived fields / validation
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
            DeepseekV4Config as _BaseTextConfig,
        )
    except Exception:
        logger.warning(
            "transformers has no DeepseekV4Config; V4.1 text config falls back to "
            "PretrainedConfig (derived fields must come from config.json)."
        )
        _BaseTextConfig = PretrainedConfig

    class DeepseekV41TextConfig(_BaseTextConfig):  # type: ignore[misc, valid-type]
        model_type = "deepseek_v41_text"
        # CSA2 source-layer routing
        kv_source_layer_ids: Optional[list] = None
        index_source_layer_ids: Optional[list] = None
        candidate_source_layer_id: Optional[int] = None
        candidate_topk_blocks: Optional[int] = None
        candidate_block_size: Optional[int] = None

        def __init__(self, **kwargs):
            if _BaseTextConfig is not PretrainedConfig:
                # V4.1 drops the hash-MoE bootstrap.
                kwargs.setdefault("default_num_hash_layers", 0)
            compress_ratios = kwargs.pop("compress_ratios", None)
            if kwargs.get("layer_types") is None and compress_ratios is not None:
                kwargs["layer_types"] = [
                    _CSA2_RATIO_TO_LAYER_TYPE.get(ratio, "sliding_attention")
                    for ratio in compress_ratios
                ]
            super().__init__(**kwargs)
            # Keep the raw ratios: mcore needs to tell ratio 1/2 apart.
            self.compress_ratios = compress_ratios

    class DeepseekV41VisionConfig(PretrainedConfig):
        model_type = "deepseek_v41_vision"

    class DeepseekV41Config(PretrainedConfig):
        model_type = "deepseek_v41"
        sub_configs = {
            "text_config": DeepseekV41TextConfig,
            "vision_config": DeepseekV41VisionConfig,
        }

        def __init__(self, text_config=None, vision_config=None, **kwargs):
            if isinstance(text_config, dict):
                text_config = DeepseekV41TextConfig(**text_config)
            elif text_config is None and "hidden_size" in kwargs:
                # Flat (non-composite) config.json: keep the text attributes local.
                text_config = None
            if isinstance(vision_config, dict):
                vision_config = DeepseekV41VisionConfig(**vision_config)
            self.text_config = text_config
            self.vision_config = vision_config
            # Surface the text config's CSA2-derived layer_types for the shared
            # DSv4 attention path.
            self.layer_types = getattr(text_config, "layer_types", None)

            # transformers >= 5.12 turns PretrainedConfig into a strict dataclass and
            # validates mlp_layer_types against the global ('sparse', 'dense')
            # vocabulary.  DeepSeek-V4 configs carry their own 'moe' / 'hash_moe'
            # labels, and this composite subclasses PretrainedConfig directly, so the
            # validator would reject them.  Remap on the composite only and keep the
            # legacy labels readable for the Megatron side; the text config keeps its
            # native vocabulary untouched.
            _legacy_mlp_layer_types = kwargs.pop("mlp_layer_types", None)
            if _legacy_mlp_layer_types is None:
                _legacy_mlp_layer_types = getattr(text_config, "mlp_layer_types", None)
            if _legacy_mlp_layer_types is not None:
                self._v41_mlp_layer_types = list(_legacy_mlp_layer_types)
                _mlp_remap = {"moe": "sparse", "hash_moe": "sparse"}
                self.mlp_layer_types = [
                    _mlp_remap.get(layer_type, layer_type)
                    for layer_type in _legacy_mlp_layer_types
                ]

            super().__init__(**kwargs)

        def __getattr__(self, name):
            # Delegate text-model attributes (vocab_size / hidden_size / engram_*)
            # to text_config so bridge and vLLM can read them off the composite.
            try:
                return super().__getattribute__(name)
            except AttributeError:
                text_config = super().__getattribute__("text_config")
                if text_config is not None and hasattr(text_config, name):
                    value = getattr(text_config, name)
                    if name == "rope_parameters" and isinstance(value, dict):
                        # vLLM expects the flat 'compress' sub-dict.
                        return value.get("compress", value)
                    return value
                raise

    for model_type, config_cls in (
        ("deepseek_v41_text", DeepseekV41TextConfig),
        ("deepseek_v41_vision", DeepseekV41VisionConfig),
        ("deepseek_v41", DeepseekV41Config),
    ):
        try:
            AutoConfig.register(model_type, config_cls, exist_ok=True)
        except TypeError:  # older transformers without exist_ok
            try:
                AutoConfig.register(model_type, config_cls)
            except ValueError:
                pass
        logger.info("Registered HF config for model_type=%r", model_type)

    # The bridge PEFT/LoRA path instantiates a dummy HF model to enumerate
    # parameters, which needs a registered model class for the composite config.
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4ForCausalLM,
        )

        class DeepseekV41ForCausalLM(DeepseekV4ForCausalLM):
            config_class = DeepseekV41Config

            def __init__(self, config):
                super().__init__(getattr(config, "text_config", config))

        AutoModelForCausalLM.register(
            DeepseekV41Config, DeepseekV41ForCausalLM, exist_ok=True
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("skip AutoModelForCausalLM.register for deepseek_v41: %r", exc)


# Register on import: run_recipe.py pulls this module in through the recipe
# star-import, well before AutoBridge.from_hf_pretrained() reads any config.
_install_hf_config_compat()
