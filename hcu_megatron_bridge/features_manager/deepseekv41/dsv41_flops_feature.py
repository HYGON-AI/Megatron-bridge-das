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


class DSV41FlopsFeature(AbstractFeature):
    """Teach the bridge FLOPs estimator the DSv4.1 ``csa_compress_ratios`` vocabulary."""

    def __init__(self):
        super().__init__("dsv41_flops", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        pass

    def register_patches(self, patch_manager, args):
        # DSv4.1 uses csa_compress_ratios in {0, 1, 2}; the bridge FLOPs estimator only
        # understands {0, 4, 128} and raises otherwise. Add the ratio 1/2 terms.
        from ...training.flops_dsv41 import (
            num_floating_point_operations as dsv41_num_floating_point_operations,
        )

        patch_manager.register_patch(
            "megatron.bridge.training.utils.flop_utils.num_floating_point_operations",
            dsv41_num_floating_point_operations,
        )
