# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Qwen3.5-VL recipe exports.

This module re-exports AutoBridge-based pretrain config helpers for the dense
Qwen3.5-VL models (2B, 4B, 9B).
"""

from hcu_megatron.bridge.recipes.qwen35_vl.bw1000.qwen35_vl import (
    qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config,
    qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config,
    qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config,
    qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_mock_config,
    qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_config,
    qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_mock_config,
)


__all__ = [
    "qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config",
    "qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config",
    "qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config",
    "qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_mock_config",
    "qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_config",
    "qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_mock_config",
]
