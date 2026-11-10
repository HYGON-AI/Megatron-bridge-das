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
"""Patch helpers for ``MegatronModelBridge.build_conversion_tasks``.

The upstream method keeps a ``None`` entry for every Megatron parameter the
mapping registry has no HF counterpart for (it already logs
``No mapping found for megatron_param: ...`` for those).  Both consumers of that
list -- ``load_weights_hf_to_megatron`` and ``export_hf_weights`` -- only guard
against ``task.megatron_module is None`` and therefore raise::

    AttributeError: 'NoneType' object has no attribute 'megatron_module'

The wrapper below filters those entries out; it is installed through
:class:`hcu_megatron.features_manager.deepseekv41.dsv41_conversion_feature.DSV41ConversionFeature`
so that the patch follows the project's feature/patch-manager flow.
"""

from __future__ import annotations

from functools import wraps


def build_conversion_tasks_wrapper(func):
    """Drop ``None`` (unmapped-parameter) tasks from the conversion work list."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tasks = func(*args, **kwargs)
        return [task for task in tasks if task is not None]

    return wrapper
