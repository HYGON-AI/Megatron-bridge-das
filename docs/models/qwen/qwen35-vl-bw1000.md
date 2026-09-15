<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Qwen3.5-VL Dense Models on BW1000

The 2B, 4B, and 9B Qwen3.5-VL dense pretraining recipes for a BW1000 node
with eight GPUs. Each size has an Energon recipe for real data and a Mock
recipe for distributed smoke tests.

The recipes live in the `hcu_megatron` patch layer. `env/env.sh` copies the
patch layer's `recipes/__init__.py` over the upstream one at every launch, which
is how the runner resolves these names; no file under `3rdparty/` is edited in
the repository:

- Recipes: [hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen35_vl/bw1000/qwen35_vl.py](../../../hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen35_vl/bw1000/qwen35_vl.py)
- Registration: [hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/__init__.py](../../../hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/__init__.py)
- Entry point: [run_recipe.py](../../../run_recipe.py)
- Launcher: [examples/qwen/train_qwen35_vl_pretrain.sh](../../../examples/qwen/train_qwen35_vl_pretrain.sh)

## Recipe matrix

| Model | TP | PP | CP | Sequence parallel | Sequence length | Energon recipe | Mock recipe |
|---|---:|---:|---:|---|---:|---|---|
| 2B | 1 | 1 | 1 | No | 4096 | `qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config` | `qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config` |
| 4B | 2 | 1 | 1 | Yes | 4096 | `qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config` | `qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_mock_config` |
| 9B | 2 | 2 | 1 | Yes | 2048 | `qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_config` | `qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_mock_config` |

Sequence parallelism requires tensor parallelism, so 4B uses TP2/PP1 rather
than TP1/PP2; `ModelParallelConfig` rejects sequence parallelism at TP1.

Architecture values (layer count, hidden size, head count, normalization, and
so on) are read from the model's own `config.json` by `AutoBridge`. The recipes
deliberately do not restate them, so a recipe cannot silently contradict the
checkpoint it loads. `seq_length` above is the training sequence length, not the
model's maximum position count.

All six recipes use BF16 with Transformer Engine, micro/global batch size 1/32,
50 iterations, Adam with betas `0.9/0.95`, weight decay `0.1`, and a cosine
learning rate from `3e-5` to `3e-6` with one warmup iteration. Evaluation is off
(`validation.eval_iters=0`) because these are short bring-up runs. All three
towers are trained; nothing is frozen, unlike the upstream alignment-stage
recipes.

## Launcher

The launcher takes the recipe name itself, then the mode:

```bash
bash examples/qwen/train_qwen35_vl_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES] [KEY=VALUE ...]
```

`MODE` is `dryrun` or `train`. Run it with no arguments to print the six
supported recipe names. The size and the dataset backend are read back out of
the recipe name, so the launcher's topology, sequence length, and dataset
wiring always match the recipe it is starting.

`HOSTFILE` and `NODES` are optional and only matter for multi-node training. An
argument in the third position that contains `=` is treated as a runner
override rather than a hostfile, so the two forms can be mixed.

## Dry run

Builds and validates the config. Needs no GPUs and no dataset:

```bash
bash examples/qwen/train_qwen35_vl_pretrain.sh \
  qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config dryrun
```

## Mock smoke tests

Mock recipes generate deterministic synthetic image/conversation samples and
need no Energon dataset:

```bash
LOCAL_HF_PATH=/data/models/Qwen3.5-2B MOCK_NUM_BASE_EXAMPLES=100 \
  bash examples/qwen/train_qwen35_vl_pretrain.sh \
    qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config train
```

Substitute `4b` or `9b` in the recipe name for the other sizes.

## Energon training

```bash
LOCAL_HF_PATH=/data/models/Qwen3.5-4B \
DATA_PATH=/data/energon/qwen35-4b \
CHECKPOINT_PATH=/data/checkpoints/qwen35-4b \
  bash examples/qwen/train_qwen35_vl_pretrain.sh \
    qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config train
```

## Multi-node training

TP, PP, and CP are fixed per size, so extra nodes only widen data parallelism.
Pass a hostfile as the third argument, one host per line, and optionally cap the
node count with the fourth:

```bash
LOCAL_HF_PATH=/data/models/Qwen3.5-9B \
GLOBAL_BATCH_SIZE=64 \
  bash examples/qwen/train_qwen35_vl_pretrain.sh \
    qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_mock_config train ./hostfile 2
```

The first host in the file becomes the rendezvous master. The launcher writes a
`slots=1` copy of the hostfile next to itself and runs one `mpirun` process per
node, each starting its own `torchrun` for that node's GPUs; `--node_rank` is
resolved on the remote host from `OMPI_COMM_WORLD_RANK`. The remote shell
re-sources `env/env.sh`, and the recipes read their model, data, batch, and mock
settings from the environment, so those variables are forwarded with `-x`. Set
`PORT` if the nodes' ssh daemon is not on 25900.

Because data parallelism grows with the node count, `GLOBAL_BATCH_SIZE` has to
stay divisible by `MICRO_BATCH_SIZE x DP`. The launcher checks this before
launching and reports the numbers rather than letting the run fail during setup.

## Checkpoint location

The launcher exports the checkpoint directory as `LOCAL_HF_PATH`, the same name
`examples/deepseekv4/train_deepseekv4_pro_pretrain.sh` uses. That name is not
cosmetic: the HCU patch layer wraps `AutoBridge.from_hf_pretrained` and
redirects it at `LOCAL_HF_PATH`, so the weights load from there no matter what
path the recipe passes in.

That wrapper covers the weights only. A VL recipe also needs the processor and
the tokenizer, so the recipe reads `LOCAL_HF_PATH` itself for
`hf_processor_path` and `tokenizer.tokenizer_model`; without that the processor
would fall back to the hub name while the weights came from disk.

One recipe runs at a time, so the launcher keeps a single checkpoint variable
rather than one per size: set `LOCAL_HF_PATH` at the top of the script, or in
the environment, and point it at the size named in the recipe. The launcher
only checks that the directory exists; the architecture itself is read from its
`config.json`, so a mismatched pair fails later in `AutoBridge`, not here.

Called directly, the recipes resolve `LOCAL_HF_PATH`, then `MODEL_HF_PATH`,
then `QWEN35_<SIZE>_HF_PATH`, then the hub name.

## Environment knobs

The launcher accepts `GPUS`, `TRAIN_ITERS`, `GLOBAL_BATCH_SIZE`,
`MICRO_BATCH_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `PORT` (ssh port used by
`mpirun`), `LOCAL_HF_PATH`, `DATA_PATH`, and `CHECKPOINT_PATH`. Mock data is
tuned with `MOCK_PROMPT`, `MOCK_NUM_IMAGES`, `MOCK_IMAGE_WIDTH`,
`MOCK_IMAGE_HEIGHT`, `MOCK_RANDOM_SEED`, and `MOCK_NUM_BASE_EXAMPLES`.

Trailing `KEY=VALUE` arguments from the third position onward are forwarded to
the runner last, so they override the launcher's own blocks:

```bash
bash examples/qwen/train_qwen35_vl_pretrain.sh \
  qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config dryrun \
  model.tensor_model_parallel_size=2 train.train_iters=10
```

The remaining knobs also have a per-size form the recipes read as a fallback:
`QWEN35_4B_DATA_PATH`, `QWEN35_9B_CHECKPOINT_PATH`, and so on. The generic name
always wins when both are set.

Set the micro batch size through `MICRO_BATCH_SIZE`, not the runner's `-mb`
flag. `ConfigContainer` requires `EnergonDatasetConfig.micro_batch_size` to
match `train.micro_batch_size`, and the recipe sets both from this one variable;
`-mb` would move only one side and the config would be rejected.

## Running the recipe directly

The launcher is a convenience wrapper around the repository's entry point. Any
upstream flag or trailing `KEY=VALUE` override works through it:

```bash
source env/env.sh
python run_recipe.py \
  --recipe qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config \
  --mode pretrain --dry-run \
  dataset.path=/data/energon/qwen35-2b \
  checkpoint.save=/data/checkpoints/qwen35-2b
```

`env/env.sh` is required, not optional: it copies the HCU recipe index into the
upstream tree and puts `hcu_megatron_bridge/src` on `PYTHONPATH`. Without it the
recipe name will not resolve.

Pass absolute paths: the runner chdirs into `3rdparty/Megatron-Bridge` before
applying overrides.

## Tests

```bash
pytest tests/unit_tests/hcu/recipes/
```

These stub out `AutoBridge`, so they check the recipes' own decisions without
downloading a checkpoint.
