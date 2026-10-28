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


class HyperConnectionFeature(AbstractFeature):

    def __init__(self):
        super().__init__('fused_mhc', optimization_level=0)


    def register_args(self, parser: ArgumentParser):
        pass
    

    def register_patches(self, patch_manager, args):

        # megatron core -- hip fused mhc kernel
        from ...core.fusions.fused_mhc_kernels import (
            _mhc_backend_status, 
            fused_sinkhorn, 
            fused_h_aggregate, 
            fused_h_aggregate_into,
            fused_h_post_bda, 
            fused_proj_rms_compute_h,
        )

        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels._mhc_backend_status', 
                                    _mhc_backend_status)
        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels.fused_sinkhorn', 
                                    fused_sinkhorn)
        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels.fused_h_aggregate', 
                                    fused_h_aggregate)
        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels.fused_h_aggregate_into',
                                    fused_h_aggregate_into)
        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels.fused_h_post_bda',
                                    fused_h_post_bda)
        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels.fused_proj_rms_compute_h',
                                    fused_proj_rms_compute_h)
