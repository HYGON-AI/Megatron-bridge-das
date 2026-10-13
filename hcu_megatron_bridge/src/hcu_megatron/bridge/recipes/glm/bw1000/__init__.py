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

"""BW1000 recipes for GLM models."""

from hcu_megatron.bridge.recipes.glm.bw1000.glm52 import (
    glm52_pretrain_32gpu_bw1000_bf16_test_config,
)
from hcu_megatron.bridge.recipes.glm.bw1000.glm53 import (
    glm53_pretrain_32gpu_bw1000_bf16_test_config,
)


__all__ = [
    "glm52_pretrain_32gpu_bw1000_bf16_test_config",
    "glm53_pretrain_32gpu_bw1000_bf16_test_config",
]
