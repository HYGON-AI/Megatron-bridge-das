# This code was adopted from https://gitcode.com/Ascend/MindSpeed
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse

from megatron.training import get_args

from .features_manager import ADAPTOR_FEATURES
from .patch_utils import MegatronPatchesManager
from .training.arguments import destroy_adaptor_args, get_adaptor_args


def patch_features():
    adaptor_args = get_adaptor_args()

    for feature in ADAPTOR_FEATURES:
        if (
            (getattr(adaptor_args, feature.feature_name, None) and feature.optimization_level == 2)
            or feature.default_patches
        ):  
            try:
                feature.register_patches(MegatronPatchesManager, adaptor_args)
            except Exception as e:
                from .utils import print_only_rank0
                print_only_rank0(f"[Warning] some feature has not been patched because {e!r}")

    MegatronPatchesManager.apply_patches()


def repatch(patch_adaptor_args, patch_megatron_args):
    destroy_adaptor_args()
    MegatronPatchesManager.remove_patches()

    adaptor_args = get_adaptor_args()
    for k, v in patch_adaptor_args.items():
        setattr(adaptor_args, k, v)

    megatron_args = get_args()
    for k, v in patch_megatron_args.items():
        setattr(megatron_args, k, v)

    patch_features()


patch_features()