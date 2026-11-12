# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
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
"""BW1100 text-only recipes for Qwen3.5 (dense and MoE sizes).

``LOCAL_HF_PATH`` lets the bw1100 launch scripts point every recipe at an
on-disk HF snapshot instead of hitting the hub; it falls back to the hub id for
ad-hoc/local use where the env var isn't set.

Only deployment choices live in the per-size spec below.  Architecture values
(layer count, hidden size, expert count, ...) are read from the model's own
``config.json`` by ``AutoBridge`` and are deliberately not restated, so a recipe
can never silently contradict the checkpoint it loads.
"""


from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


@dataclass(frozen=True)
class _Qwen35TextSpec:
    """Deployment topology for one Qwen3.5 text size on BW1100."""

    size: str
    hf_default: str
    # Nodes are derived as world_gpus / 8; sizes above one node keep their
    # tp/pp/ep and scale data parallelism across hosts.
    world_gpus: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    expert_model_parallel_size: int
    global_batch_size: int
    # MoE models take the router/permute/dispatcher policy and disable the
    # DDP/comm overlaps that were not validated for them; dense models reject
    # any expert_model_parallel_size > 1.
    moe: bool


# Dense sizes keep TP*PP within one 8-GPU node (0.8B/2B TP1, 4B/9B TP2,
# 27B TP4*PP2). 35B-A3B and 122B-A10B are MoE and spread experts over EP.
_QWEN35_TEXT_SPECS = {
    "0.8b": _Qwen35TextSpec(
        size="0.8b", hf_default="Qwen/Qwen3.5-0.8B", world_gpus=8,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1,
        global_batch_size=128, moe=False,
    ),
    "2b": _Qwen35TextSpec(
        size="2b", hf_default="Qwen/Qwen3.5-2B", world_gpus=8,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1,
        global_batch_size=128, moe=False,
    ),
    "4b": _Qwen35TextSpec(
        size="4b", hf_default="Qwen/Qwen3.5-4B", world_gpus=8,
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, expert_model_parallel_size=1,
        global_batch_size=128, moe=False,
    ),
    "9b": _Qwen35TextSpec(
        size="9b", hf_default="Qwen/Qwen3.5-9B", world_gpus=8,
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, expert_model_parallel_size=1,
        global_batch_size=128, moe=False,
    ),
    "27b": _Qwen35TextSpec(
        size="27b", hf_default="Qwen/Qwen3.5-27B", world_gpus=8,
        tensor_model_parallel_size=2, pipeline_model_parallel_size=2, expert_model_parallel_size=1,
        global_batch_size=128, moe=False,
    ),
    "35b_a3b": _Qwen35TextSpec(
        size="35b_a3b", hf_default="Qwen/Qwen3.5-35B-A3B", world_gpus=16,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=2, expert_model_parallel_size=8,
        global_batch_size=512, moe=True,
    ),
    "122b_a10b": _Qwen35TextSpec(
        size="122b_a10b", hf_default="Qwen/Qwen3.5-122B-A10B", world_gpus=32,
        tensor_model_parallel_size=4, pipeline_model_parallel_size=2, expert_model_parallel_size=4,
        global_batch_size=512, moe=True,
    ),
    # 397B-A17B: 397B params need ~4.4 TB for weights+grads+fp32 Adam states, so
    # 32 GPUs would put ~139 GB on every card with no room for activations.
    # 64 GPUs brings that to ~69 GB/card. TP4 divides the 32 attention heads,
    # PP4 divides the 60 layers, and EP8 divides both 512 experts and DP*TP=16.
    "397b_a17b": _Qwen35TextSpec(
        size="397b_a17b", hf_default="Qwen/Qwen3.5-397B-A17B", world_gpus=64,
        tensor_model_parallel_size=4, pipeline_model_parallel_size=4, expert_model_parallel_size=8,
        global_batch_size=512, moe=True,
    ),
}


def _hf_path(spec: _Qwen35TextSpec) -> str:
    """Resolve the HuggingFace checkpoint this size loads from."""
    return os.environ.get("LOCAL_HF_PATH") or spec.hf_default


def _qwen35_text_pretrain_config(spec: _Qwen35TextSpec) -> ConfigContainer:
    """Build the shared BW1100 text-only pretraining config for one size.

    Every public recipe below is this builder with a fixed spec, so model,
    optimizer, parallelism and BW1100 policy stay identical across sizes apart
    from the per-size topology in the spec.
    """
    cfg = _pretrain_common()

    hf_path = _hf_path(spec)
    text_config = AutoConfig.from_pretrained(hf_path).text_config
    # The nested text config intentionally omits ``architectures``. AutoBridge
    # needs it to select the registered causal-LM bridge instead of the VLM.
    text_config.architectures = ["Qwen3_5MoeForCausalLM" if spec.moe else "Qwen3_5ForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    cfg.model.tensor_model_parallel_size = spec.tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = spec.pipeline_model_parallel_size
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = spec.expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = spec.global_batch_size
    # MBS4 is suitable for force-balanced throughput benchmarking, but OOMs
    # with learned routing. MBS1 was validated with real RP2 data on GB200.
    cfg.train.micro_batch_size = 1

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    if spec.moe:
        cfg.model.moe_router_fusion = True
        cfg.model.moe_permute_fusion = True
        cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    # Keep the library-safe native implementation instead of the performance
    # harness's TE cross-entropy path, which currently warns about stability.
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.apply_rope_fusion = True

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Fixed-length text batches can use the scopes that the VLM recipe must
    # disable for variable-length multimodal inputs.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = (
        ["attn", "moe_router", "moe_preprocess"] if spec.moe else ["attn", "mlp"]
    )
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    if spec.moe:
        # alltoall (NCCL all-to-all) instead of flex+hybridep: HybridEP is not
        # installed in this cluster's container (the flex/hybridep path dies at
        # TransformerLayer construction with "HybridEP is not installed").
        # alltoall is Megatron-Core's built-in dispatcher and only needs NCCL.
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_router_dtype = "fp32"
        cfg.model.moe_shared_expert_overlap = False
        cfg.model.moe_router_force_load_balancing = False
        cfg.model.moe_router_padding_for_fp8 = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    if spec.moe:
        cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    if not spec.moe:
        # Only the dense sizes override mixed_precision/rerun_state_machine/
        # ddp.grad_reduce_in_fp32; the MoE sizes (35B-A3B, 122B-A10B) keep
        # _pretrain_common()'s defaults for all three, matching the original
        # qwen35_text_35b_a3b recipe.
        cfg.mixed_precision = bf16_mixed()
        cfg.mixed_precision.grad_reduce_in_fp32 = False
        cfg.ddp.grad_reduce_in_fp32 = False
        cfg.rerun_state_machine.check_for_nan_in_loss = False

    # MoE sizes disable the DDP overlap and nan checks that were only validated
    # on the dense path; dense sizes keep them on.
    cfg.ddp.overlap_grad_reduce = not spec.moe
    cfg.ddp.overlap_param_gather = not spec.moe
    cfg.ddp.check_for_nan_in_grad = spec.moe
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=spec.moe,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    ) if spec.moe else CommOverlapConfig(tp_comm_overlap=False)
    return cfg


def qwen35_text_0_8b_pretrain_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3.5-0.8B text-only pretraining, 8 BW1100 GPUs (TP1/PP1, DP8)."""
    return _qwen35_text_pretrain_config(_QWEN35_TEXT_SPECS["0.8b"])


def qwen35_text_2b_pretrain_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3.5-2B text-only pretraining, 8 BW1100 GPUs (TP1/PP1, DP8)."""
    return _qwen35_text_pretrain_config(_QWEN35_TEXT_SPECS["2b"])


def qwen35_text_4b_pretrain_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3.5-4B text-only pretraining, 8 BW1100 GPUs (TP2/PP1, DP4)."""
    return _qwen35_text_pretrain_config(_QWEN35_TEXT_SPECS["4b"])


def qwen35_text_9b_pretrain_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3.5-9B text-only pretraining, 8 BW1100 GPUs (TP2/PP1, DP4)."""
    return _qwen35_text_pretrain_config(_QWEN35_TEXT_SPECS["9b"])


def qwen35_text_27b_pretrain_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3.5-27B text-only pretraining, 8 BW1100 GPUs (TP2/PP2)."""
    return _qwen35_text_pretrain_config(_QWEN35_TEXT_SPECS["27b"])


def qwen35_text_35b_a3b_pretrain_16gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3.5-35B-A3B text-only pretraining, 2 nodes x 8 BW1100 GPUs (TP1/PP2/EP8)."""
    return _qwen35_text_pretrain_config(_QWEN35_TEXT_SPECS["35b_a3b"])


def qwen35_text_122b_a10b_pretrain_32gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3.5-122B-A10B text-only pretraining, 4 nodes x 8 BW1100 GPUs (TP4/PP2/EP4)."""
    return _qwen35_text_pretrain_config(_QWEN35_TEXT_SPECS["122b_a10b"])


def qwen35_text_397b_a17b_pretrain_64gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3.5-397B-A17B text-only pretraining, 8 nodes x 8 BW1100 GPUs (TP4/PP4/EP8)."""
    return _qwen35_text_pretrain_config(_QWEN35_TEXT_SPECS["397b_a17b"])
