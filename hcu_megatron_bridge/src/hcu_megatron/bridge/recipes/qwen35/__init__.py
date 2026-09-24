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

"""Qwen35 recipe exports.

This module re-exports AutoBridge-based pretrain config helpers for Qwen35.
"""

from .qwen35 import (
    qwen35_text_35b_a3b_pretrain_32gpu_bw1000_bf16_config,
)


__all__ = [
    "qwen35_text_35b_a3b_pretrain_32gpu_bw1000_bf16_config",
]
