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
"""BW1101 recipes for DeepSeek V4 Flash."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_squad_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from hcu_megatron.bridge.models.deepseek.deepseek_v4_bridge import deepseek_v4_supports_bw_fused_kernels


DEEPSEEK_V4_FLASH_HF_PATH = "deepseek-ai/DeepSeek-V4-Flash-0731"


# ---------------------------------------------------------------------------
# Parameter-Efficient Fine-Tuning (PEFT)
# ---------------------------------------------------------------------------


def deepseek_v4_flash_peft_32gpu_bw1101_bf16_test_config() -> ConfigContainer:
    """Return the DeepSeek-V4-Flash BW1101 peft base config.

    Recommended BW1101 baseline: TP=1, PP=4, EP=8, CP=1.
    """
    cfg = _sft_common()
    cfg.model = AutoBridge.from_hf_pretrained(DEEPSEEK_V4_FLASH_HF_PATH).to_megatron_provider(
        load_weights=False
    )

    # --- parallelism (DSv4 hybrid attention requires TP=1) ---
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.params_dtype = torch.bfloat16
    cfg.model.seq_length = 4096

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = deepseek_v4_supports_bw_fused_kernels()
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False

    # --- MoE ---
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.calculate_per_token_loss = True

    # --- MTP enabled ---
    if getattr(cfg.model, "mtp_num_layers", None):
        cfg.model.mtp_loss_scaling_factor = 0.1

    # --- tokenizer / dataset (real HF tokenizer; SBHD / unpacked) ---
    cfg.tokenizer.tokenizer_model = DEEPSEEK_V4_FLASH_HF_PATH
    cfg.dataset = default_squad_config(seq_length=4096, enable_offline_packing=False)

    # --- robustness defaults ---
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.dist.enable_megatron_core_experimental = True

    cfg.peft = LoRA(
        target_modules=list(
            [
                "linear_q_down_proj",
                "linear_q_up_proj",
                "linear_kv_proj",
                "linear_proj",
                "linear_fc1",
                "linear_fc2",
            ]
        ),
        dim=32,
        alpha=32,
        dropout=0.0,
        share_expert_adapters=False,
    )
    cfg.optimizer.lr = 1.0e-4
    cfg.optimizer.min_lr = 0.0

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg