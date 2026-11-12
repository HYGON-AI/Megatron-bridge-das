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

"""Qwen3.5-VL dense pretraining recipes for one BW1000 node (8 GPUs).

These recipes live in the ``hcu_megatron`` patch layer.  ``env/env.sh`` copies
the patch layer's ``recipes/__init__.py`` over the upstream one at every launch,
which is how ``run_recipe.py`` resolves these names, so nothing under
``3rdparty/`` is modified in the repository.

Topologies on 8 GPUs: 2B is TP1/PP1 (DP8), 4B is TP2/PP1/SP (DP4), and 9B is
TP2/PP2/SP (DP2).  Each size can run against an Energon dataset or synthetic
VLM data; all model, optimizer, and BW1000 settings stay in one shared builder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import (
    EnergonDatasetConfig,
    MockVLMSFTDatasetConfig,
    QwenVLEnergonTaskEncoderConfig,
)
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


@dataclass(frozen=True)
class _Qwen35VLSpec:
    """Deployment topology for one dense Qwen3.5-VL size on a BW1000 node.

    Only deployment choices live here.  Architecture values (layer count,
    hidden size, head count, and so on) are read from the model's own
    ``config.json`` by ``AutoBridge`` and are deliberately not restated, so a
    recipe can never silently contradict the checkpoint it loads.
    """

    size: str
    hf_default: str
    seq_length: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    sequence_parallel: bool
    # MoE sizes spread experts over EP; dense sizes must keep this at 1 or
    # Megatron-Core rejects the config.
    expert_model_parallel_size: int = 1
    moe: bool = False


_QWEN35_VL_SPECS = {
    "2B": _Qwen35VLSpec(
        size="2B",
        hf_default="Qwen/Qwen3.5-2B",
        seq_length=4096,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
    ),
    # Sequence parallelism requires tensor parallelism, so 4B uses TP2 rather
    # than the TP1/PP2/SP combination, which ModelParallelConfig rejects.
    "4B": _Qwen35VLSpec(
        size="4B",
        hf_default="Qwen/Qwen3.5-4B",
        seq_length=4096,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        sequence_parallel=True,
    ),
    "9B": _Qwen35VLSpec(
        size="9B",
        hf_default="Qwen/Qwen3.5-9B",
        seq_length=2048,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        sequence_parallel=True,
    ),
    # Larger sizes need more than one 8-GPU node; TP*PP (dense) or PP*EP (MoE)
    # fixes the minimum node count and extra nodes only widen data parallelism.
    "27B": _Qwen35VLSpec(
        size="27B",
        hf_default="Qwen/Qwen3.5-27B",
        seq_length=4096,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        sequence_parallel=True,
    ),
    "35B-A3B": _Qwen35VLSpec(
        size="35B-A3B",
        hf_default="Qwen/Qwen3.5-35B-A3B",
        seq_length=4096,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        # SP needs TP>1, so it stays off for this TP1 topology.
        sequence_parallel=False,
        expert_model_parallel_size=8,
        moe=True,
    ),
    "122B-A10B": _Qwen35VLSpec(
        size="122B-A10B",
        hf_default="Qwen/Qwen3.5-122B-A10B",
        seq_length=4096,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        sequence_parallel=True,
        expert_model_parallel_size=4,
        moe=True,
    ),
    "397B-A17B": _Qwen35VLSpec(
        size="397B-A17B",
        hf_default="Qwen/Qwen3.5-397B-A17B",
        seq_length=4096,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        sequence_parallel=True,
        expert_model_parallel_size=8,
        moe=True,
    ),
}

def _model_env(spec: _Qwen35VLSpec, name: str, default: str | None = None) -> str | None:
    """Read a generic variable, then the model-specific compatibility name."""

    return os.environ.get(name) or os.environ.get(f"QWEN35_{spec.size}_{name}", default)


def _hf_path(spec: _Qwen35VLSpec) -> str:
    """Resolve the HuggingFace checkpoint this size loads from.

    ``LOCAL_HF_PATH`` comes first because the HCU patch layer already redirects
    ``AutoBridge.from_hf_pretrained`` at it.  Reading it here as well keeps the
    processor and the tokenizer on the same local directory as the weights;
    without that they would fall back to the hub name below.
    """

    return (
        os.environ.get("LOCAL_HF_PATH")
        or os.environ.get("MODEL_HF_PATH")
        or os.environ.get(f"QWEN35_{spec.size}_HF_PATH")
        or spec.hf_default
    )


def _env_int(spec: _Qwen35VLSpec, name: str, default: int) -> int:
    """Read an integer knob, falling back to ``default`` when unset or blank."""

    raw_value = _model_env(spec, name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def _qwen35_vl_pretrain_config(
    spec: _Qwen35VLSpec,
    *,
    dataset_backend: Literal["energon", "mock"],
) -> ConfigContainer:
    """Build a shared BW1000 configuration for one dense Qwen3.5-VL size.

    ``dataset_backend`` changes only the dataset config object.  Keeping that
    choice in this private helper gives every public recipe the same model,
    optimizer, parallelism, and BW1000 policy.
    """
    cfg = _pretrain_common()

    hf_path = _hf_path(spec)

    # ConfigContainer rejects an Energon dataset whose micro batch size
    # disagrees with train.micro_batch_size, so one knob drives both.
    micro_batch_size = _env_int(spec, "MICRO_BATCH_SIZE", 1)

    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = spec.seq_length
    cfg.model.tensor_model_parallel_size = spec.tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = spec.pipeline_model_parallel_size
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = spec.sequence_parallel
    cfg.model.expert_model_parallel_size = spec.expert_model_parallel_size
    cfg.model.pipeline_dtype = torch.bfloat16 if spec.pipeline_model_parallel_size > 1 else None

    # Full pretraining: no frozen submodules.  Upstream's alignment-stage
    # recipes freeze the language and vision towers; that is a different task.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # BW1000/ROCm compatibility policy.
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.use_te_rng_tracker = False
    cfg.model.bias_activation_fusion = False
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    if spec.moe:
        # alltoall is Megatron-Core's built-in NCCL dispatcher; the flex/hybridep
        # path needs the HybridEP package, which this cluster does not ship.
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_router_dtype = "fp32"
    # AttnBackend is an enum, and the config compares it by identity, so a
    # plain "auto" string would never match and would break backend dispatch.
    cfg.model.attention_backend = AttnBackend.auto
    cfg.model.cross_entropy_loss_fusion = False
    cfg.rng.te_rng_tracker = False
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.train.train_iters = _env_int(spec, "TRAIN_ITERS", 50)
    cfg.train.global_batch_size = _env_int(spec, "GLOBAL_BATCH_SIZE", 32)
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.manual_gc = False

    # Evaluation is off by default: these are short bring-up runs, and an
    # Energon dataset would otherwise need a prepared validation split.
    cfg.validation.eval_interval = 1000
    cfg.validation.eval_iters = 0

    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=1,
        lr_decay_iters=cfg.train.train_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        max_lr=3.0e-5,
        min_lr=3.0e-6,
        clip_grad=1.0,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
    )
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    if dataset_backend == "energon":
        # The dataset type and Qwen task encoder belong to the recipe; the data
        # location comes from DATA_PATH or a ``dataset.path=...`` override.
        cfg.dataset = EnergonDatasetConfig(
            path=_model_env(spec, "DATA_PATH"),
            seq_length=spec.seq_length,
            micro_batch_size=micro_batch_size,
            num_workers=2,
            task_encoder=QwenVLEnergonTaskEncoderConfig(hf_processor_path=hf_path),
            dataloader_type="external",
            do_validation=False,
            do_test=False,
            enable_in_batch_packing=False,
            pad_to_max_length=True,
        )
    elif dataset_backend == "mock":
        # Synthetic image/conversation data for model and distributed smoke
        # tests.  The class name says SFT because it emits chat-shaped
        # examples; the caller still selects the pretrain loop.
        cfg.dataset = MockVLMSFTDatasetConfig(
            seq_length=spec.seq_length,
            hf_processor_path=hf_path,
            prompt=_model_env(spec, "MOCK_PROMPT") or "Describe this image.",
            random_seed=_env_int(spec, "MOCK_RANDOM_SEED", 0),
            image_size=(
                _env_int(spec, "MOCK_IMAGE_WIDTH", 256),
                _env_int(spec, "MOCK_IMAGE_HEIGHT", 256),
            ),
            num_images=_env_int(spec, "MOCK_NUM_IMAGES", 1),
            num_base_examples=_env_int(spec, "MOCK_NUM_BASE_EXAMPLES", 1000),
            num_workers=1,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
            pad_to_max_length=True,
        )
    else:
        raise ValueError(f"Unsupported Qwen3.5-VL {spec.size} dataset backend: {dataset_backend!r}")

    cfg.tokenizer.tokenizer_model = hf_path
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = True
    cfg.ddp.use_distributed_optimizer = True

    cfg.checkpoint.save_interval = 1000
    cfg.checkpoint.ckpt_format = "torch_dist"
    checkpoint_path = _model_env(spec, "CHECKPOINT_PATH")
    if checkpoint_path:
        cfg.checkpoint.save = checkpoint_path
        cfg.checkpoint.load = checkpoint_path
        cfg.logger.tensorboard_dir = os.path.join(checkpoint_path, "tensorboard")
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "GPU_MAX_HW_QUEUES": 4,
    }
    cfg.mixed_precision = bf16_mixed()
    return cfg


def qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config() -> ConfigContainer:
    """Return the Energon-backed Qwen3.5-VL 2B BW1000 recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["2B"], dataset_backend="energon")


def qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Return the synthetic-data Qwen3.5-VL 2B BW1000 smoke-test recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["2B"], dataset_backend="mock")


def qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config() -> ConfigContainer:
    """Return the Energon-backed Qwen3.5-VL 4B BW1000 recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["4B"], dataset_backend="energon")


def qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Return the synthetic-data Qwen3.5-VL 4B BW1000 smoke-test recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["4B"], dataset_backend="mock")


def qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_config() -> ConfigContainer:
    """Return the Energon-backed Qwen3.5-VL 9B BW1000 recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["9B"], dataset_backend="energon")


def qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Return the synthetic-data Qwen3.5-VL 9B BW1000 smoke-test recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["9B"], dataset_backend="mock")

def qwen35_vl_27b_pretrain_8gpu_bw1000_bf16_config() -> ConfigContainer:
    """Return the Energon-backed Qwen3.5-VL 27B BW1000 recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["27B"], dataset_backend="energon")


def qwen35_vl_27b_pretrain_8gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Return the synthetic-data Qwen3.5-VL 27B BW1000 smoke-test recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["27B"], dataset_backend="mock")


def qwen35_vl_35b_a3b_pretrain_16gpu_bw1000_bf16_config() -> ConfigContainer:
    """Return the Energon-backed Qwen3.5-VL 35B-A3B BW1000 recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["35B-A3B"], dataset_backend="energon")


def qwen35_vl_35b_a3b_pretrain_16gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Return the synthetic-data Qwen3.5-VL 35B-A3B BW1000 smoke-test recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["35B-A3B"], dataset_backend="mock")


def qwen35_vl_122b_a10b_pretrain_8gpu_bw1000_bf16_config() -> ConfigContainer:
    """Return the Energon-backed Qwen3.5-VL 122B-A10B BW1000 recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["122B-A10B"], dataset_backend="energon")


def qwen35_vl_122b_a10b_pretrain_8gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Return the synthetic-data Qwen3.5-VL 122B-A10B BW1000 smoke-test recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["122B-A10B"], dataset_backend="mock")


def qwen35_vl_397b_a17b_pretrain_32gpu_bw1000_bf16_config() -> ConfigContainer:
    """Return the Energon-backed Qwen3.5-VL 397B-A17B BW1000 recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["397B-A17B"], dataset_backend="energon")


def qwen35_vl_397b_a17b_pretrain_32gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Return the synthetic-data Qwen3.5-VL 397B-A17B BW1000 smoke-test recipe."""

    return _qwen35_vl_pretrain_config(_QWEN35_VL_SPECS["397B-A17B"], dataset_backend="mock")


__all__ = [
    "qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config",
    "qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config",
    "qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config",
    "qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_mock_config",
    "qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_config",
    "qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_mock_config",
    "qwen35_vl_27b_pretrain_8gpu_bw1000_bf16_config",
    "qwen35_vl_27b_pretrain_8gpu_bw1000_bf16_mock_config",
    "qwen35_vl_35b_a3b_pretrain_16gpu_bw1000_bf16_config",
    "qwen35_vl_35b_a3b_pretrain_16gpu_bw1000_bf16_mock_config",
    "qwen35_vl_122b_a10b_pretrain_8gpu_bw1000_bf16_config",
    "qwen35_vl_122b_a10b_pretrain_8gpu_bw1000_bf16_mock_config",
    "qwen35_vl_397b_a17b_pretrain_32gpu_bw1000_bf16_config",
    "qwen35_vl_397b_a17b_pretrain_32gpu_bw1000_bf16_mock_config",
]
