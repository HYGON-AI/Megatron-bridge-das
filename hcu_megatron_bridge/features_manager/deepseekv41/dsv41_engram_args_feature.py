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


class DSV41EngramArgsFeature(AbstractFeature):
    """Let ``model.engram_*`` CLI overrides through the struct-mode config check."""

    def __init__(self):
        super().__init__("dsv41_engram_args", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        pass

    def register_patches(self, patch_manager, args):
        from megatron.bridge.training.utils import omegaconf_utils

        from ...engram_overrides import wrap_process_config

        patch_manager.register_patch(
            "megatron.bridge.training.utils.omegaconf_utils.process_config_with_overrides",
            wrap_process_config(omegaconf_utils.process_config_with_overrides),
        )
