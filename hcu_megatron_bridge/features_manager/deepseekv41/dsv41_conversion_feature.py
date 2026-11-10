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

from argparse import ArgumentParser

from ..feature import AbstractFeature


class DSV41ConversionFeature(AbstractFeature):
    """Skip conversion tasks for Megatron parameters that have no HF mapping.

    ``build_conversion_tasks`` keeps a ``None`` entry for every unmapped
    parameter (DSv4.1 Engram tensors and a few optional CSA2 weights are
    imported by dedicated loaders instead of the generic registry), and the
    generic import/export loops dereference ``task.megatron_module`` without
    checking the task itself.
    """

    def __init__(self):
        super().__init__("dsv41_conversion", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        pass

    def register_patches(self, patch_manager, args):
        from hcu_megatron.bridge.models.conversion.model_bridge import (
            build_conversion_tasks_wrapper,
        )

        patch_manager.register_patch(
            "megatron.bridge.models.conversion.model_bridge.MegatronModelBridge.build_conversion_tasks",
            build_conversion_tasks_wrapper,
            apply_wrapper=True,
        )
