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
from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


# Sequence length constants
SEQUENCE_LENGTH_32K: int = 32768
SEQUENCE_LENGTH_128K: int = 131072


# Gemma3 models
def gemma3_1b_pretrain_8gpu_bw1000_bf16_config() -> ConfigContainer:
    """Return a pre-training config for Gemma3 1B.

    Default parallelism: TP=1, PP=1, seq_length=32K
    """
    cfg = _pretrain_common()

    cfg.model = Gemma3ModelProvider(
        num_layers=26,
        hidden_size=1152,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=256,
        ffn_hidden_size=6912,
        window_size=512,
        rope_scaling_factor=1.0,
        seq_length=SEQUENCE_LENGTH_32K,
        vocab_size=262_144,
    )

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = SEQUENCE_LENGTH_32K

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = SEQUENCE_LENGTH_32K  # 32768

    # Pipeline split settings (for larger models with PP > 1)
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False

    # Training config (DIFFERENT from _pretrain_common)
    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None  # None means auto selection
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # Gemma3 uses native

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - uses "bf16_mixed" from _pretrain_common
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Optimizer settings (commented - enable for precision-aware optimizer)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# ============================================================================
# SFT Configs
# ============================================================================


def _adjust_gemma3_vocab_size(model_cfg, hf_path: str):
    """Adjust vocab size for Gemma3 (model vocab < tokenizer vocab).

    Note: This requires HuggingFace authentication for Gemma3 models.
    If the tokenizer cannot be loaded, the vocab size adjustment is skipped.
    """
    if hasattr(model_cfg, "vocab_size") and hf_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        if len(tokenizer) > model_cfg.vocab_size:
            model_cfg.vocab_size = len(tokenizer)




__all__ = [
    "gemma3_1b_pretrain_8gpu_bw1000_bf16_config",
]
