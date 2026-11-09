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
"""BW1000 recipes for DeepSeek V4 Flash."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.deepseek.deepseek_v4_bridge import set_deepseek_v4_pipeline_model_parallel_layout
from megatron.bridge.recipes.common import _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_squad_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_muon_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed
from hcu_megatron.bridge.models.deepseek.deepseek_v4_bridge import deepseek_v4_supports_bw_fused_kernels


DEEPSEEK_V4_FLASH_HF_PATH = "deepseek-ai/DeepSeek-V4-Flash-0731"


# ---------------------------------------------------------------------------
# Pretrain
# ---------------------------------------------------------------------------


def deepseek_v4_flash_pretrain_32gpu_bw1000_bf16_test_config() -> ConfigContainer:
    """Return the DeepSeek-V4-Flash BW1000 pre-training base config.

    Recommended BW1000 baseline: TP=1, PP=2, EP=16, CP=1.
    """
    use_fused_mhc = deepseek_v4_supports_bw_fused_kernels()
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(DEEPSEEK_V4_FLASH_HF_PATH, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )

    # ========== test config ==========
    cfg.model.num_layers = 4
    cfg.model.mtp_num_layers = None
    cfg.model.moe_layer_freq = 1
    cfg.model.moe_n_hash_layers = 1
    cfg.model.csa_compress_ratios = [0, 4, 0, 128]
    # ========== test config ==========

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.params_dtype = torch.bfloat16

    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = use_fused_mhc
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.tokenizer.make_vocab_size_divisible_by = cfg.model.make_vocab_size_divisible_by
    cfg.tokenizer.tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
    cfg.tokenizer.rank = 0

    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.dataset.dataloader_type = "single"

    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5
    cfg.validation.eval_interval = 2000
    cfg.validation.eval_iters = 32

    cfg.logger.log_interval = 10
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False
    cfg.dist.enable_megatron_core_experimental = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_megatron_fsdp = False

    opt_cfg, scheduler_cfg = distributed_muon_with_cosine_annealing(
        muon_momentum=0.95,
        muon_use_nesterov=True,
        muon_scale_mode="unit_rms_norm",
        muon_fp32_matmul_prec="highest",
        muon_num_ns_steps=5,
        muon_extra_scale_factor=0.2,
        lr_warmup_iters=2000,
        lr_decay_iters=cfg.train.train_iters,
        max_lr=2.7e-4,
        min_lr=2.7e-5,
        weight_decay=0.1,
        clip_grad=1.0,
    )
    # DSv4 Muon uses non-layer-wise optimizer dispatch.
    opt_cfg.optimizer = "muon"
    opt_cfg.adam_beta1 = 0.9
    opt_cfg.adam_beta2 = 0.95
    opt_cfg.adam_eps = 1e-20

    if hasattr(opt_cfg, "muon_coefficient_type"):
        opt_cfg.muon_coefficient_type = "quintic"

    scheduler_cfg.start_weight_decay = 0.1
    scheduler_cfg.end_weight_decay = 0.1
    scheduler_cfg.weight_decay_incr_style = "constant"

    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# ---------------------------------------------------------------------------
# Supervised fine-tuning (SFT)
# ---------------------------------------------------------------------------


def deepseek_v4_flash_sft_32gpu_bw1000_bf16_test_config() -> ConfigContainer:
    """Return the DeepSeek-V4-Flash BW1000 sft base config.
        
    Recommended BW1000 baseline: TP=1, PP=2, EP=16, CP=1.
    """
    cfg = _sft_common()
    cfg.model = AutoBridge.from_hf_pretrained(DEEPSEEK_V4_FLASH_HF_PATH, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )

    # ========== test config ==========
    cfg.model.num_layers = 6
    cfg.model.mtp_num_layers = None
    cfg.model.moe_layer_freq = 1
    cfg.model.csa_compress_ratios = [0, 0, 4, 128, 4, 128]
    # ========== test config ==========

    # --- parallelism (DSv4 hybrid attention requires TP=1) ---
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.expert_model_parallel_size = 16
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
    cfg.model.recompute_method =  None
    cfg.model.recompute_num_layers = None
    cfg.model.cuda_graph_impl = "none"

    # --- MTP enabled ---
    if getattr(cfg.model, "mtp_num_layers", None):
        cfg.model.mtp_loss_scaling_factor = 0.1

    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

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
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg