#!/bin/bash
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
#
# Thin launcher for train_qwen35_vl_pretrain.sh.
#
# The training script takes its model path, dataset path, and batch knobs from
# the environment rather than from arguments -- a trailing KEY=VALUE is a
# ConfigContainer override forwarded to run_recipe.py, so `DATA_PATH=... ` in
# that position never reaches the shell.  This file is the place to keep those
# variables so they do not have to be retyped, or mistyped, on every run.

set -euo pipefail
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )

# ===================================== User Settings =====================================
# Every assignment below is a default: an exported variable of the same name
# already in the environment wins, so a one-off run can override any of them
# without editing this file.

# Local Qwen3.5-VL checkpoint.  The size here must match the size in RECIPE:
# the recipe picks the topology, this directory's own config.json picks the
# architecture, and AutoBridge fails late and confusingly when they disagree.
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/public/home/zhaoyu/zy/models/Qwen3.5-VL-4B}

# Prepared Energon dataset -- the --output-dir of prepare_mantis_energon.py,
# not the intermediate source directory.  Ignored by the _mock_config recipes.
export DATA_PATH=${DATA_PATH:-/public/home/zhaoyu/zy/dataset/coco-energon}

# Training knobs.  GLOBAL_BATCH_SIZE must stay divisible by
# MICRO_BATCH_SIZE x DP, where DP = (NNODES x GPUS) / (TP x PP x CP).
export TRAIN_ITERS=${TRAIN_ITERS:-50}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export GPUS=${GPUS:-8}

# Checkpoints, TensorBoard, and train logs.  Left unset, the training script
# puts them next to itself in <recipe>_tp<TP>_pp<PP>_cp<CP>; uncomment to
# follow the ${WORKSPACE}/results layout documented in README_coco_energon.md.
# export CHECKPOINT_PATH=${CHECKPOINT_PATH:-${WORKSPACE:-/workspace}/results/qwen35_vl}

# ===================================== Launch =====================================
TRAIN_SCRIPT="${CURRENT_DIR}/train_qwen35_vl_pretrain.sh"

usage() {
    cat <<USAGE
Usage: $0 [RECIPE] [MODE] [HOSTFILE] [NODES] [KEY=VALUE ...]

  RECIPE  default: qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config
  MODE    dryrun | train   (default: train)

Recipes:
  qwen35_vl_{2b,4b,9b}_pretrain_8gpu_bw1000_bf16_config        Energon, needs DATA_PATH
  qwen35_vl_{2b,4b,9b}_pretrain_8gpu_bw1000_bf16_mock_config   synthetic data, no DATA_PATH

Trailing KEY=VALUE arguments are recipe config overrides applied last, for
example train.lr=1e-5 or logger.log_interval=1.  Paths belong in the User
Settings block above, or in the environment -- not here.

Examples:
  $0                                                      # 4B energon, train
  $0 qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config dryrun  # resolve config only
  $0 qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config train
  DATA_PATH=/other/dataset $0                             # override for one run
USAGE
}

case "${1:-}" in
    -h|--help|help) usage; exit 0 ;;
esac

RECIPE=${1:-qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config}
MODE=${2:-train}
# Shifted one at a time: `shift 2` with a single argument present fails without
# shifting, which would send RECIPE back through as a config override.
(( $# >= 1 )) && shift || true
(( $# >= 1 )) && shift || true
EXTRA_ARGS=("$@")

if [[ ! -f ${TRAIN_SCRIPT} ]]; then
    echo "ERROR: ${TRAIN_SCRIPT} not found."
    exit 1
fi

# The training script validates RECIPE, MODE, and both paths, and prints the
# run summary, so nothing is re-checked here.  exec keeps its exit code and
# leaves Ctrl-C going straight to torchrun.
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    exec bash "${TRAIN_SCRIPT}" "${RECIPE}" "${MODE}" "${EXTRA_ARGS[@]}"
else
    exec bash "${TRAIN_SCRIPT}" "${RECIPE}" "${MODE}"
fi
