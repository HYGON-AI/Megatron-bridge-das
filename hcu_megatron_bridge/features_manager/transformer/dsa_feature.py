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
from ...training.arguments import remove_original_params


class DSAFeature(AbstractFeature):

    def __init__(self):
        super().__init__('fused_dsa', optimization_level=0)


    def register_args(self, parser: ArgumentParser):
        remove_original_params(parser, ["dsa_kernel_backend"])
        group = parser.add_argument_group(title='extra training args')
    
        group.add_argument('--dsa-kernel-backend', type=str, choices=['none', 'tilelang', 'cudnn', 'hip'],
                        default=None,
                        help='Fused DSA kernel backend. When omitted, resolves to "cudnn" for '
                        'experimental_attention_variant="dsv4_hybrid" launches and to "none" '
                        'otherwise.',)
    

    def register_patches(self, patch_manager, args):

        # megatron core -- hip fused dsa kernel
        from ...core.utils import (
            _VALID_DSA_KERNEL_BACKENDS, 
            _validate_dsa_kernel_backend_dependencies,
        )
        from ...core.transformer.experimental_attention_variant.dsa_kernels import (
            _BACKEND_MODULE_NAME_BY_BACKEND, 
            _get_dsa_kernel_backend,
        )
        from ...core.transformer.experimental_attention_variant.dsa_hip_kernels import (
            _dsa_fwd_flash_mla,
            _ensure_dsa_namespace_hip,  
        )
        from ...core.transformer.experimental_attention_variant.csa_utils.fused_compressor import (
            _MINIMUM_COMPUTE_CAPABILITY_MAJOR, 
            _get_frontend,
        )
        from ...core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
            _csa_fwd_flash_mla,
            _ensure_dsa_namespace,
        )

        patch_manager.register_patch('megatron.core.utils._VALID_DSA_KERNEL_BACKENDS',
                                    _VALID_DSA_KERNEL_BACKENDS)
        patch_manager.register_patch('megatron.core.utils._validate_dsa_kernel_backend_dependencies',
                                    _validate_dsa_kernel_backend_dependencies)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.dsa_kernels._BACKEND_MODULE_NAME_BY_BACKEND',
                                    _BACKEND_MODULE_NAME_BY_BACKEND)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.dsa_kernels._get_dsa_kernel_backend',
                                    _get_dsa_kernel_backend)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels._dsa_fwd_flash_mla',
                                    _dsa_fwd_flash_mla)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels._ensure_dsa_namespace',
                                    _ensure_dsa_namespace_hip)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.csa_utils.fused_compressor._MINIMUM_COMPUTE_CAPABILITY_MAJOR',
                                    _MINIMUM_COMPUTE_CAPABILITY_MAJOR)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.csa_utils.fused_compressor._get_frontend',
                                    _get_frontend)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention._csa_fwd_flash_mla',
                                    _csa_fwd_flash_mla)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention._ensure_dsa_namespace',
                                    _ensure_dsa_namespace)


        # V4.1 hard-codes dsa_kernel_backend in ("none", "cudnn"); HCU adds 'hip'.
        # Run the original validator with an allowed placeholder so every other V4.1
        # rule is still enforced, then restore the real backend for kernel dispatch.
        from megatron.core.transformer.transformer_config import TransformerConfig

        _original_validate_dsv41_config = TransformerConfig._validate_dsv41_config
        _V41_ALLOWED_BACKENDS = ("none", "cudnn")

        def _validate_dsv41_config_hcu(self):
            backend = self.dsa_kernel_backend
            if backend in _V41_ALLOWED_BACKENDS:
                _original_validate_dsv41_config(self)
                return
            # 'none' short-circuits the fused-kernel dependency probe that
            # _validate_dsv41_config performs internally (line ~4381), while
            # still satisfying its backend-name whitelist.
            self.dsa_kernel_backend = "none"
            try:
                _original_validate_dsv41_config(self)
            finally:
                self.dsa_kernel_backend = backend

        patch_manager.register_patch('megatron.core.transformer.transformer_config.TransformerConfig._validate_dsv41_config',
                                    _validate_dsv41_config_hcu)
