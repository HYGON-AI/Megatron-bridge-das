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

"""Tests for the HCU Qwen3.5-VL BW1000 recipes.

``AutoBridge`` is replaced by a stub so these tests describe the recipe's own
decisions without downloading a checkpoint.  Architecture values are therefore
deliberately not asserted: the recipe leaves them to the model's ``config.json``.
"""

from types import SimpleNamespace

import pytest

from megatron.bridge.data.builders import EnergonDatasetConfig, MockVLMSFTDatasetConfig
from megatron.bridge.training.config import ConfigContainer


MODULE_PATH = "hcu_megatron.bridge.recipes.qwen35_vl.bw1000.qwen35_vl"


class _FakeAutoBridge:
    """Stand-in for AutoBridge that yields a bare, writable provider."""

    @staticmethod
    def from_hf_pretrained(path):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights=False):
        return SimpleNamespace()


_SIZES = ("2B", "4B", "9B")

# The checkpoint location has a chain of its own: LOCAL_HF_PATH is the name the
# launcher exports and the patch layer redirects AutoBridge with, MODEL_HF_PATH
# is the older generic name, and the per-size fallback is QWEN35_<SIZE>_HF_PATH
# rather than QWEN35_<SIZE>_LOCAL_HF_PATH.
_HF_PATH_KNOBS = ("LOCAL_HF_PATH", "MODEL_HF_PATH", *(f"QWEN35_{size}_HF_PATH" for size in _SIZES))

# Every other knob the recipe reads, cleared so an ambient shell cannot change
# results.  Each of these also has a per-size QWEN35_<SIZE>_<NAME> form.
_ENV_KNOBS = (
    "DATA_PATH",
    "CHECKPOINT_PATH",
    "TRAIN_ITERS",
    "GLOBAL_BATCH_SIZE",
    "MICRO_BATCH_SIZE",
    "MOCK_PROMPT",
    "MOCK_RANDOM_SEED",
    "MOCK_IMAGE_WIDTH",
    "MOCK_IMAGE_HEIGHT",
    "MOCK_NUM_IMAGES",
    "MOCK_NUM_BASE_EXAMPLES",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove generic and per-size knobs left over from the caller's shell."""
    for name in _HF_PATH_KNOBS:
        monkeypatch.delenv(name, raising=False)
    for name in _ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)
        for size in _SIZES:
            monkeypatch.delenv(f"QWEN35_{size}_{name}", raising=False)


@pytest.fixture()
def bw1000(monkeypatch):
    """Import the recipe module with AutoBridge stubbed out."""
    module = pytest.importorskip(MODULE_PATH)
    monkeypatch.setattr(module, "AutoBridge", _FakeAutoBridge)
    return module


def test_defaults_suit_a_short_bring_up_run(bw1000, monkeypatch):
    """Unset knobs give a short run with evaluation disabled."""
    monkeypatch.setenv("LOCAL_HF_PATH", "/models/qwen35-2b")

    cfg = bw1000.qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config()

    assert cfg.train.train_iters == 50
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.validation.eval_iters == 0
    assert cfg.optimizer.lr == 3.0e-5
    assert cfg.optimizer.min_lr == 3.0e-6
    # Unset DATA_PATH leaves the location to a dataset.path=... override.
    assert cfg.dataset.path is None

@pytest.mark.parametrize(
    ("size", "tensor_parallel", "pipeline_parallel", "sequence_parallel", "seq_length"),
    [
        ("2b", 1, 1, False, 4096),
        ("4b", 2, 1, True, 4096),
        ("9b", 2, 2, True, 2048),
    ],
)
def test_energon_recipe_topology(
    bw1000, monkeypatch, size, tensor_parallel, pipeline_parallel, sequence_parallel, seq_length
):
    """Each size declares its documented BW1000 topology."""
    monkeypatch.setenv("LOCAL_HF_PATH", f"/models/qwen35-{size}")
    monkeypatch.delenv("DATA_PATH", raising=False)

    cfg = getattr(bw1000, f"qwen35_vl_{size}_pretrain_8gpu_bw1000_bf16_config")()

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model.tensor_model_parallel_size == tensor_parallel
    assert cfg.model.pipeline_model_parallel_size == pipeline_parallel
    assert cfg.model.sequence_parallel is sequence_parallel
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.seq_length == seq_length
    assert isinstance(cfg.dataset, EnergonDatasetConfig)
    assert cfg.dataset.seq_length == seq_length
    assert cfg.dataset.task_encoder.hf_processor_path == f"/models/qwen35-{size}"


def test_sequence_parallelism_always_has_tensor_parallelism(bw1000, monkeypatch):
    """ModelParallelConfig rejects sequence parallelism without TP > 1."""
    monkeypatch.setenv("LOCAL_HF_PATH", "/models/qwen35")

    for size in ("2b", "4b", "9b"):
        cfg = getattr(bw1000, f"qwen35_vl_{size}_pretrain_8gpu_bw1000_bf16_config")()
        if cfg.model.sequence_parallel:
            assert cfg.model.tensor_model_parallel_size > 1


def test_pipeline_dtype_set_only_when_pipelining(bw1000, monkeypatch):
    """pipeline_dtype is meaningful only for a multi-stage pipeline."""
    import torch

    monkeypatch.setenv("LOCAL_HF_PATH", "/models/qwen35")

    single_stage = bw1000.qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config()
    multi_stage = bw1000.qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_config()

    assert single_stage.model.pipeline_dtype is None
    assert multi_stage.model.pipeline_dtype is torch.bfloat16


def test_micro_batch_size_agrees_across_train_and_dataset(bw1000, monkeypatch):
    """One knob drives both sides, which ConfigContainer requires for Energon."""
    monkeypatch.setenv("LOCAL_HF_PATH", "/models/qwen35-2b")
    monkeypatch.setenv("MICRO_BATCH_SIZE", "4")

    cfg = bw1000.qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config()

    assert cfg.train.micro_batch_size == 4
    assert cfg.dataset.micro_batch_size == 4


def test_mock_recipe_uses_synthetic_dataset(bw1000, monkeypatch):
    """The mock backend swaps the dataset and reads its own knobs."""
    monkeypatch.setenv("LOCAL_HF_PATH", "/models/qwen35-2b")
    monkeypatch.setenv("MOCK_PROMPT", "Summarize the image.")
    monkeypatch.setenv("MOCK_NUM_BASE_EXAMPLES", "7")
    monkeypatch.setenv("MOCK_NUM_IMAGES", "2")

    cfg = bw1000.qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config()

    assert isinstance(cfg.dataset, MockVLMSFTDatasetConfig)
    assert not isinstance(cfg.dataset, EnergonDatasetConfig)
    assert cfg.dataset.hf_processor_path == "/models/qwen35-2b"
    assert cfg.dataset.prompt == "Summarize the image."
    assert cfg.dataset.num_base_examples == 7
    assert cfg.dataset.num_images == 2


def test_model_specific_env_var_is_honoured(bw1000, monkeypatch):
    """The per-size QWEN35_<SIZE>_HF_PATH name still works."""
    monkeypatch.setenv("QWEN35_4B_HF_PATH", "/models/qwen35-4b")

    cfg = bw1000.qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config()

    assert cfg.dataset.task_encoder.hf_processor_path == "/models/qwen35-4b"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        # LOCAL_HF_PATH is what the launcher exports and what the patch layer
        # redirects AutoBridge.from_hf_pretrained at, so it has to win: the
        # processor and the tokenizer must load from the same directory as the
        # weights.
        (
            {
                "LOCAL_HF_PATH": "/models/local",
                "MODEL_HF_PATH": "/models/generic",
                "QWEN35_4B_HF_PATH": "/models/qwen35-4b",
            },
            "/models/local",
        ),
        ({"MODEL_HF_PATH": "/models/generic", "QWEN35_4B_HF_PATH": "/models/qwen35-4b"}, "/models/generic"),
        ({"QWEN35_4B_HF_PATH": "/models/qwen35-4b"}, "/models/qwen35-4b"),
        # Nothing set at all falls back to the spec's hub name.
        ({}, "Qwen/Qwen3.5-4B"),
    ],
)
def test_hf_path_precedence(bw1000, monkeypatch, env, expected):
    """LOCAL_HF_PATH beats MODEL_HF_PATH, which beats the per-size fallback."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    cfg = bw1000.qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config()

    assert cfg.dataset.task_encoder.hf_processor_path == expected
    # The tokenizer has to follow the same directory, not just the processor.
    assert cfg.tokenizer.tokenizer_model == expected


def test_bw1000_compatibility_settings(bw1000, monkeypatch):
    """The BW1000/ROCm policy is applied to every recipe."""
    from megatron.core.transformer.enums import AttnBackend

    monkeypatch.setenv("LOCAL_HF_PATH", "/models/qwen35-2b")

    cfg = bw1000.qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config()

    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.use_te_rng_tracker is False
    assert cfg.model.bias_activation_fusion is False
    assert cfg.model.cross_entropy_loss_fusion is False
    # An enum, not the string "auto": the config compares backends by identity.
    assert cfg.model.attention_backend is AttnBackend.auto
    assert cfg.rng.te_rng_tracker is False
    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.env_vars["GPU_MAX_HW_QUEUES"] == 4


def test_full_pretraining_freezes_nothing(bw1000, monkeypatch):
    """Pretraining trains every tower, unlike the alignment-stage recipes."""
    monkeypatch.setenv("LOCAL_HF_PATH", "/models/qwen35-2b")

    cfg = bw1000.qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config()

    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_unknown_dataset_backend_is_rejected(bw1000, monkeypatch):
    """An unsupported backend fails loudly rather than silently."""
    monkeypatch.setenv("LOCAL_HF_PATH", "/models/qwen35-2b")
    spec = bw1000._QWEN35_VL_DENSE_SPECS["2B"]

    with pytest.raises(ValueError, match="dataset backend"):
        bw1000._qwen35_vl_dense_pretrain_config(spec, dataset_backend="parquet")
