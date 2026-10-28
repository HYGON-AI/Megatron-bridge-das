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


class ProfilerFeature(AbstractFeature):

    def __init__(self):
        super().__init__('profiler', optimization_level=0)


    def register_args(self, parser: ArgumentParser):
        pass
    

    def register_patches(self, patch_manager, args):

        import importlib.util

        from hcu_megatron_bridge.utils import profiler_annotate_patch
        from megatron.core.fusions.fused_mhc_kernels import is_cutile_available

        # `fused_mhc` is a HIP/ROCm-only package; on NVIDIA it isn't installed,
        # and Patch.parse_path would raise ModuleNotFoundError on those entries.
        has_fused_mhc = importlib.util.find_spec('fused_mhc') is not None
        has_cutile = is_cutile_available()

        targets = {
            # unfused mhc
            'megatron.core.transformer.hyper_connection.SinkhornKnopp.forward': 'HyperConnection::native_sinkhorn_forward',
            'megatron.core.transformer.hyper_connection.SinkhornKnopp.backward': 'HyperConnection::native_sinkhorn_backward',
            'megatron.core.transformer.hyper_connection.native_h_aggregate': 'HyperConnection::native_h_aggregate',
            'megatron.core.fusions.fused_mhc_kernels._torch_h_aggregate_bwd': 'HyperConnection::torch_h_aggregate_bwd',
            'megatron.core.transformer.hyper_connection.native_h_post_bda': 'HyperConnection::native_h_post_bda',
            'megatron.core.fusions.fused_mhc_kernels._torch_h_post_bda_bwd': 'HyperConnection::torch_h_post_bda_bwd',
            'megatron.core.fusions.fused_mhc_kernels._torch_proj_rms_compute_h': 'HyperConnection::torch_proj_rms_compute_h',
            'megatron.core.transformer.hyper_connection.HyperConnectionModule._projection_and_get_norm': 'HyperConnection::projection_and_get_norm',
            'megatron.core.transformer.hyper_connection.HyperConnectionModule._compute_h': 'HyperConnection::compute_h',

            # unfused dsa
            'megatron.core.transformer.experimental_attention_variant.csa.unfused_compressed_sparse_attn': 'CompressedSparseAttention::unfused_compressed_sparse_attn',

            # fused mhc --- hip
            'fused_mhc.FusedSinkhorn.forward': 'HipFusedSinkhorn::forward',
            'fused_mhc.FusedSinkhorn.backward': 'HipFusedSinkhorn::backward',
            'fused_mhc.FusedHAggregate.forward': 'HipFusedHAggregate::forward',
            'fused_mhc.FusedHAggregate.backward': 'HipFusedHAggregate::backward',
            'fused_mhc.FusedHAggregateInto.forward': 'HipFusedHAggregateInto::forward',
            'fused_mhc.FusedHAggregateInto.backward': 'HipFusedHAggregateInto::backward',
            'fused_mhc.FusedHPostBDA.forward': 'HipFusedHPostBDA::forward',
            'fused_mhc.FusedHPostBDA.backward': 'HipFusedHPostBDA::backward',
            'fused_mhc.FusedProjRmsComputeH.forward': 'HipFusedProjRmsComputeH::forward',
            'fused_mhc.FusedProjRmsComputeH.backward': 'HipFusedProjRmsComputeH::backward',

            # fused mhc --- triton
            'megatron.core.fusions.fused_mhc_kernels._triton_sinkhorn_fwd': 'TritonFusedSinkhorn::forward',
            'megatron.core.fusions.fused_mhc_kernels._triton_sinkhorn_bwd': 'TritonFusedSinkhorn::backward',
            'megatron.core.fusions.fused_mhc_kernels._triton_h_aggregate_fwd': 'TritonFusedHAggregate::forward',
            'megatron.core.fusions.fused_mhc_kernels._triton_h_post_bda_fwd': 'TritonFusedHPostBDA::forward',
            'megatron.core.fusions.fused_mhc_kernels._triton_h_post_bda_bwd': 'TritonFusedHPostBDA::backward',

            # fused mhc --- cutile
            'megatron.core.fusions.fused_mhc_kernels._cutile_sinkhorn_fwd': 'CutileFusedSinkhorn::forward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_sinkhorn_bwd': 'CutileFusedSinkhorn::backward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_h_aggregate_fwd': 'CutileFusedHAggregate::forward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_h_aggregate_bwd': 'CutileFusedHAggregate::backward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_h_post_bda_fwd': 'CutileFusedHPostBDA::forward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_h_post_bda_bwd': 'CutileFusedHPostBDA::backward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_proj_rms_compute_h_fwd': 'CutileFusedProjRmsComputeH::forward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_fused_compute_h_proj_rms_bwd': 'CutileFusedProjRmsComputeH::backward',

            # fused dsa
            'megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention.CSASparseAttnFunc.forward': 'CSASparseAttnFunc::fused_compressed_sparse_attn_forward',
            'megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention.CSASparseAttnFunc.backward': 'CSASparseAttnFunc::fused_compressed_sparse_attn_backward',
            'megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention.FusedCSAIndexerSparseAttnFunc.forward': 'FusedCSAIndexerSparseAttnFunc::fused_compressed_sparse_attn_forward',
            'megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention.FusedCSAIndexerSparseAttnFunc.backward': 'FusedCSAIndexerSparseAttnFunc::fused_compressed_sparse_attn_backward',
        }

        for path, message in targets.items():
            if path.startswith('fused_mhc.') and not has_fused_mhc:
                continue
            if '_cutile' in path and not has_cutile:
                continue

            patch_manager.register_patch(path, profiler_annotate_patch(path, message), apply_wrapper=True)