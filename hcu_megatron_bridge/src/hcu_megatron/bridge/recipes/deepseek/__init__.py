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

"""DeepSeek recipe exports.

This module re-exports AutoBridge-based pretrain config helpers for DeepSeek
models V4.
"""

from .deepseek_v4 import (
    deepseek_v4_pro_pretrain_bw1000_bf16_config,
    deepseek_v4_pro_pretrain_256gpu_bw1000_bf16_test_config,
)


__all__ = [
    "deepseek_v4_pro_pretrain_bw1000_bf16_config",
    "deepseek_v4_pro_pretrain_256gpu_bw1000_bf16_test_config",
]
