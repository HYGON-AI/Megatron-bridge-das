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


from .basic.megatron_basic import MegatronBasicFeature
from .deepseekv4.deepseekv4_feature import DeepSeekV4Feature
from .deepseekv41.dsv41_engram_args_feature import DSV41EngramArgsFeature
from .deepseekv41.dsv41_flops_feature import DSV41FlopsFeature
from .profiler.profiler_feature import ProfilerFeature
from .transformer.dsa_feature import DSAFeature
from .transformer.hyper_connection_feature import HyperConnectionFeature


ADAPTOR_FEATURES = [
    MegatronBasicFeature(),
    DeepSeekV4Feature(),
    DSV41EngramArgsFeature(),
    DSV41FlopsFeature(),
    ProfilerFeature(),
    DSAFeature(),
    HyperConnectionFeature(),
]