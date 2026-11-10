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
"""BW1300 recipes for DeepSeek V4.1 (V4 backbone + Engram conditional memory).

The V4.1 model is assembled by :class:`DeepSeekV41Bridge`, which derives the
Engram configuration from the HF config and attaches the Engram composition
point to every decoder layer spec.  These recipes only choose parallelism and
the training knobs that Engram cares about:

* ``engram_layer_ids`` decides which global layers own memory tables, so the
  hybrid/attention layout must keep those layers on the attention (not Mamba)
  slots.
* Engram issues its own variable-split all-to-all, so the MoE dispatcher can
  stay on any backend; the hash-routed MoE config is inherited from V4.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_muon_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed
from hcu_megatron.bridge.models.deepseek.deepseek_v4_bridge import deepseek_v4_supports_bw_fused_kernels
from hcu_megatron.bridge.models.deepseek.deepseek_v41_bridge import has_engram


DEEPSEEK_V41_FLASH_HF_PATH = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def _require_engram() -> None:
    if not has_engram():
        raise RuntimeError(
            "This recipe needs Megatron-Core with the Engram extension. Replace "
            "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM with the DeepSeek-V4.1 "
            "branch (dsv41-pr7224-engram-local) before running it."
        )


# ---------------------------------------------------------------------------
# Pretrain
# ---------------------------------------------------------------------------


def deepseek_v41_flash_pretrain_16gpu_bw1300_bf16_test_config() -> ConfigContainer:
    """Return the DeepSeek-V4.1-Flash BW1300 pre-training base config.
    
    Recommended BW1300 baseline: TP=1, PP=1, EP=16, CP=1.
    """
    _require_engram()
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(DEEPSEEK_V41_FLASH_HF_PATH, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )

    # ========== test config==========
    cfg.model.moe_layer_freq = 1
    cfg.model.csa2_kv_source_layers = [1]
    cfg.model.csa2_index_source_layers = [1]
    cfg.model.csa2_candidate_source_layer = 1
    # ========== test config==========

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.params_dtype = torch.bfloat16

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.apply_dsa_kernel_fusion = None
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = deepseek_v4_supports_bw_fused_kernels()
    cfg.model.dsa_indexer_loss_coeff = 1e-2
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.dsa_indexer_rotate_activation = False

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.mscale = 0
    cfg.model.mscale_all_dim = 0

    # Engram rejects activation recomputation / CUDA graphs / VPP.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.model.cuda_graph_impl = "none"

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
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "DEEP_EP_NORMAL_MNVL": "1",
    }
    return cfg

