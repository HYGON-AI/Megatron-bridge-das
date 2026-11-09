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

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


MINIMAX_M3_HF_PATH = "MiniMaxAI/MiniMax-M3"


# ---------------------------------------------------------------------------
# Pretrain
# ---------------------------------------------------------------------------


def minimax_m3_pretrain_16gpu_bw1000_bf16_test_config() -> ConfigContainer:
    
    cfg = _pretrain_common()
    
    vlm_provider = AutoBridge.from_hf_pretrained(MINIMAX_M3_HF_PATH, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )
    cfg.model = vlm_provider.to_text_provider()

    # ========== test config ==========
    cfg.model.num_layers = 2
    cfg.model.moe_layer_freq = 1
    # ========== test config ==========

    # Parallelism
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096
    cfg.model.params_dtype = torch.bfloat16

    # 60 layers split evenly across 4 stages
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_permute_fusion_into_hybridep = False
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving: recompute the MoE activation function only
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.cuda_graph_impl = "none"

    # Tokenizer - uses NullTokenizer by default (no HF tokenizer download needed)
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8

    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5
    cfg.validation.eval_interval = 2000

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_interval = 10
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False

    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )

    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


__all__ = [
    "minimax_m3_pretrain_16gpu_bw1000_bf16_test_config",
]