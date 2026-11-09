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
"""BW1000 recipes for GLM-5.2."""

from __future__ import annotations
import os

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_tulu3_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_GLM52_MODEL_ID = "zai-org/GLM-5.2"
_GLM52_MODEL_REVISION = "4d67f66cc64d3219133b767c253b2ad1425c6c88"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# Pretrain
# ---------------------------------------------------------------------------


def glm52_pretrain_32gpu_bw1000_bf16_test_config() -> ConfigContainer:
    """GLM-5.2 bounded pretraining on 32 bw1000 GPUs."""
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(_GLM52_MODEL_ID).to_megatron_provider(
        load_weights=False
    )

    # ========== test config ==========
    cfg.model.num_layers = 4
    cfg.model.mtp_num_layers = None
    cfg.model.moe_layer_freq = ([0]*3+[1]*1)
    cfg.model.pipeline_model_parallel_layout = "Et*2|t*2L"
    # ========== test config ==========

    cfg.tokenizer.tokenizer_model = _GLM52_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _GLM52_MODEL_REVISION}

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 13
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.microbatch_group_size_per_vp_stage = 16

    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.calculate_per_token_loss = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True

    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1

    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.dataset.random_seed = 1234
    cfg.rng.seed = 1234
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=40,
        max_lr=3e-4,
        min_lr=3e-5,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None
    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS}
    return cfg

# ---------------------------------------------------------------------------
# Supervised Fine-Tuning
# ---------------------------------------------------------------------------


def glm52_sft_32gpu_bw1000_bf16_test_config() -> ConfigContainer:
    """GLM-5.2 bounded full SFT on 32 BW1000 GPUs."""
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_GLM52_MODEL_ID).to_megatron_provider(
        load_weights=False
    )

    # ========== test config ==========
    cfg.model.num_layers = 4
    cfg.model.mtp_num_layers = None
    cfg.model.moe_layer_freq = ([0]*3+[1]*1)
    cfg.model.pipeline_model_parallel_layout = "Et*2|t*2L"
    # ========== test config ==========

    cfg.tokenizer.tokenizer_model = _GLM52_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _GLM52_MODEL_REVISION}
    cfg.model.seq_length = 8192
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.microbatch_group_size_per_vp_stage = 6

    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.calculate_per_token_loss = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.deallocate_pipeline_outputs = True
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.apply_rope_fusion = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1

    packing_alignment = 2 * cfg.model.context_parallel_size
    cfg.dataset = default_tulu3_config(
        seq_length=cfg.model.seq_length,
        enable_offline_packing=True,
        pad_seq_to_mult=packing_alignment,
    )
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = cfg.model.seq_length - packing_alignment
    # HybridEP needs a fixed token width; CUDA graphs are disabled, so cu_seqlens can remain dynamic.
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "USE_MNNVL": 1,
    }
    return cfg


__all__ = [
    "glm52_pretrain_64gpu_bw1000_bf16_test_config",
    "glm52_sft_32gpu_bw1000_bf16_test_config",
]
