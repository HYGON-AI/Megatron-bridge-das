#!/bin/bash

set -euo pipefail
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
BRIDGE_ENV_SCRIPT="${BRIDGE_ENV_SCRIPT:-${MEGATRON_PATH}/env/env.sh}" # cluster env script (NCCL/IB vars, PYTHONPATH, HCU recipe registration); override with --env BRIDGE_ENV_SCRIPT=<path>

# default env
PORT=${PORT:-25900}
source ${BRIDGE_ENV_SCRIPT}

# env.sh does not cover Megatron-Energon; add it when a local checkout is
# present.  Remembered separately because the remote shell of a multi-node run
# re-sources env.sh and would otherwise lose it.
ENERGON_SRC="${MEGATRON_PATH}/3rdparty/Megatron-Energon/src"
EXTRA_PYTHONPATH=""
if [[ -d "${ENERGON_SRC}" ]]; then
    export PYTHONPATH="${PYTHONPATH}:${ENERGON_SRC}"
    EXTRA_PYTHONPATH="${ENERGON_SRC}"
fi

# One recipe runs at a time, so one checkpoint variable is enough.  The hcu
# patch layer reads LOCAL_HF_PATH to redirect AutoBridge.from_hf_pretrained at
# a local directory, and the recipe reads it too so the processor and the
# tokenizer load from the same place.  Point it at the size being trained.
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/Qwen3.5-VL-2B}

# ===================================== Launch Mode =====================================
RECIPE=${1:-}
MODE=${2:-}
shift 2 2>/dev/null || true

# [HOSTFILE] and [NODES] keep deepseek's positions, and anything left over is a
# KEY=VALUE override forwarded to the runner.  A third argument containing '='
# is an override, not a hostfile, so both forms can be mixed freely.
HOSTFILE=""
NODES=""
if [[ $# -gt 0 && ${1} != *=* ]]; then
    HOSTFILE=${1}
    shift
    if [[ $# -gt 0 && ${1} =~ ^[0-9]+$ ]]; then
        NODES=${1}
        shift
    fi
fi
EXTRA_ARGS=("$@")

RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

# The six BW1000 recipes published by hcu_megatron.bridge.recipes.qwen35_vl
SUPPORTED_RECIPES=(
    qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config
    qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_mock_config
    qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_mock_config
)

# Validate RECIPE argument
if [[ -z ${RECIPE} ]]; then
    echo -e "${RED}ERROR: missing recipe argument.${RESET}"
    echo -e "${YELLOW}Supported recipes:${RESET}"
    printf "  - %s\n" "${SUPPORTED_RECIPES[@]}"
    echo ""
    echo "Usage: $0 <RECIPE> <MODE> [HOSTFILE] [NODES] [KEY=VALUE ...]"
    echo "  MODE: dryrun | train"
    exit 1
fi

if ! printf '%s\n' "${SUPPORTED_RECIPES[@]}" | grep -qx "${RECIPE}"; then
    echo -e "${RED}ERROR: unsupported recipe '${RECIPE}'.${RESET}"
    echo -e "${YELLOW}Supported recipes:${RESET}"
    printf "  - %s\n" "${SUPPORTED_RECIPES[@]}"
    exit 1
fi

case "${MODE}" in
    dryrun|train) ;;
    *)
        echo -e "${RED}ERROR: MODE must be 'dryrun' or 'train', got: '${MODE}'${RESET}"
        exit 1
        ;;
esac

# Size and dataset backend are carried by the recipe name itself.
SIZE=$(echo "${RECIPE}" | sed -n 's/^qwen35_vl_\([0-9]\+b\)_.*/\1/p')
SIZE_UPPER=$(echo "${SIZE}" | tr '[:lower:]' '[:upper:]')
if [[ ${RECIPE} == *_mock_config ]]; then
    BACKEND=mock
else
    BACKEND=energon
fi

# ===================================== Multi-node Topology =====================================
if [[ ${RECIPE} == *bw1300* ]]; then
    GPUS=${GPUS:-4}
else
    GPUS=${GPUS:-8}
fi
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}

# NNODES comes from the hostfile line count, or the NODES argument, or 1.
# TP/PP are fixed per size, so extra nodes only widen data parallelism.
if [[ -n ${HOSTFILE} ]]; then
    [[ -f ${HOSTFILE} ]] || { echo -e "${RED}ERROR: ${HOSTFILE} not found.${RESET}"; exit 1; }
    HOSTFILE=$(cd $(dirname ${HOSTFILE}) && pwd)/$(basename ${HOSTFILE})
    # grep -c exits 1 when it counts nothing, which under set -e would abort
    # the script with no message at all, so absorb that and report it here.
    AVAILABLE_NODES=$(grep -cve '^[[:space:]]*$' "${HOSTFILE}" || true)
    if (( AVAILABLE_NODES == 0 )); then
        echo -e "${RED}ERROR: ${HOSTFILE} lists no hosts.${RESET}"
        exit 1
    fi
    NNODES=${NODES:-${AVAILABLE_NODES}}
    if (( NNODES > AVAILABLE_NODES )); then
        echo -e "${RED}ERROR: node count ${NNODES} exceeds ${AVAILABLE_NODES} host(s) in ${HOSTFILE}.${RESET}"
        exit 1
    fi
    # First host is the rendezvous master; mpirun places local rank 0 there.
    MASTER_ADDR=$(grep -ve '^[[:space:]]*$' ${HOSTFILE} | sed -n 1p | awk '{print $1}')
else
    NNODES=${NODES:-1}
    if (( NNODES > 1 )); then
        echo -e "${RED}ERROR: multi-node run requires a hostfile.${RESET}"
        exit 1
    fi
fi
WORLD_GPUS=$((NNODES * GPUS))
# NODES accepts any digits and GPUS comes from the environment, so zero on
# either side is reachable; without this the divisibility check below divides
# by zero and set -e aborts with only bash's own message.
if (( WORLD_GPUS < 1 )); then
    echo -e "${RED}ERROR: ${NNODES} node(s) x ${GPUS} GPU(s) leaves nothing to run on.${RESET}"
    exit 1
fi

# ===================================== Model and Data Paths =====================================
# The recipe name picks the topology and LOCAL_HF_PATH picks the weights, so
# the two have to agree.  Only existence is checked here; the architecture
# itself is read from the directory's own config.json by AutoBridge.
if [[ ! -d "${LOCAL_HF_PATH}" ]]; then
    echo -e "${RED}ERROR: model directory not found: ${LOCAL_HF_PATH}${RESET}"
    echo "Set LOCAL_HF_PATH to a local Qwen3.5-VL ${SIZE_UPPER} checkpoint."
    exit 1
fi

DATA_PATH_VAR="QWEN35_${SIZE_UPPER}_DATA_PATH"
DATA_PATH="${DATA_PATH:-${!DATA_PATH_VAR:-}}"
if [[ ${BACKEND} == energon && -z ${DATA_PATH} ]]; then
    echo -e "${RED}ERROR: ${RECIPE} needs a prepared Energon dataset.${RESET}"
    echo "Set ${DATA_PATH_VAR} (or DATA_PATH), or use the matching _mock_config recipe."
    exit 1
fi
# Exported even when empty so that mpirun's -x always finds it; the recipe
# treats an empty value the same as unset.
export DATA_PATH

# ===================================== Data Config =====================================
# 9B trains at 2048 to keep activations within one node; 2B and 4B use 4096.
if [[ ${SIZE} == 9b ]]; then
    SEQ_LENGTH=2048
else
    SEQ_LENGTH=4096
fi

DATA_CONFIG=(
    dataset.seq_length=${SEQ_LENGTH}
)
if [[ ${BACKEND} == energon ]]; then
    DATA_CONFIG+=(dataset.path=${DATA_PATH})
fi

# ===================================== Train Config =====================================
TRAIN_ITERS=${TRAIN_ITERS:-50}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
# EnergonDatasetConfig.micro_batch_size must equal train.micro_batch_size, and
# the recipe sets both from this one variable; passing -mb would move only one.
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_ITERS GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE

TRAIN_CONFIG=(
    train.train_iters=${TRAIN_ITERS}
    train.global_batch_size=${GLOBAL_BATCH_SIZE}
    train.micro_batch_size=${MICRO_BATCH_SIZE}
    validation.eval_iters=0
)

# ===================================== Parallel Config =====================================
# These mirror the recipe's own topology. Sequence parallelism requires tensor
# parallelism, so 4B is TP2/PP1 rather than the TP1/PP2 that ModelParallelConfig
# rejects.
case "${SIZE}" in
    2b) TP=1; PP=1; CP=1; SP=false ;;
    4b) TP=2; PP=1; CP=1; SP=true  ;;
    9b) TP=2; PP=2; CP=1; SP=true  ;;
esac

PARALLEL_CONFIG=(
    model.tensor_model_parallel_size=${TP}
    model.pipeline_model_parallel_size=${PP}
    model.context_parallel_size=${CP}
)

# Catch an unusable topology here rather than several minutes into startup.
MODEL_PARALLEL=$((TP * PP * CP))
if (( WORLD_GPUS % MODEL_PARALLEL != 0 )); then
    echo -e "${RED}ERROR: ${WORLD_GPUS} GPUs is not divisible by TP*PP*CP=${MODEL_PARALLEL}.${RESET}"
    exit 1
fi
DP=$((WORLD_GPUS / MODEL_PARALLEL))
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) != 0 )); then
    echo -e "${RED}ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by"
    echo -e "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} x DP=${DP} on ${NNODES} node(s).${RESET}"
    exit 1
fi

# ===================================== Model Config =====================================
MODEL_CONFIG=(
    model.seq_length=${SEQ_LENGTH}
    model.sequence_parallel=${SP}
)

# ===================================== Mock Data Config =====================================
# Ignored by the energon backend; the recipe reads these from the environment.
MOCK_PROMPT=${MOCK_PROMPT:-Describe this image.}
MOCK_NUM_IMAGES=${MOCK_NUM_IMAGES:-1}
MOCK_NUM_BASE_EXAMPLES=${MOCK_NUM_BASE_EXAMPLES:-1000}
MOCK_RANDOM_SEED=${MOCK_RANDOM_SEED:-0}
MOCK_IMAGE_WIDTH=${MOCK_IMAGE_WIDTH:-256}
MOCK_IMAGE_HEIGHT=${MOCK_IMAGE_HEIGHT:-256}
export MOCK_PROMPT MOCK_NUM_IMAGES MOCK_NUM_BASE_EXAMPLES MOCK_RANDOM_SEED MOCK_IMAGE_WIDTH MOCK_IMAGE_HEIGHT

# ===================================== Checkpoint & Logging Config =====================================
SAVE_DIR=${CHECKPOINT_PATH:-${CURRENT_DIR}/${RECIPE}_tp${TP}_pp${PP}_cp${CP}}
CONFIG_DUMP=${SAVE_DIR}/${RECIPE}.yaml

CKPT_AND_LOGGING_CONFIG=(
    checkpoint.load=${SAVE_DIR}
    checkpoint.save=${SAVE_DIR}
    checkpoint.save_interval=100
    logger.log_interval=10
    logger.log_throughput=true
    logger.tensorboard_dir=${SAVE_DIR}/tb_logs
    logger.save_config_filepath=${CONFIG_DUMP}
)

# ===================================== Summary =====================================
mkdir -p ${SAVE_DIR}/train_logs
echo "======================================"
echo "Qwen3.5-VL ${SIZE_UPPER} pretrain"
echo "======================================"
echo "Mode:        ${MODE}"
echo "Recipe:      ${RECIPE}"
echo "Nodes:       ${NNODES}${HOSTFILE:+ (${HOSTFILE})}"
echo "GPUs:        ${GPUS}/node, ${WORLD_GPUS} total   (TP=${TP}, PP=${PP}, CP=${CP}, SP=${SP}, DP=${DP})"
echo "Iters:       ${TRAIN_ITERS}"
echo "GBS/MBS:     ${GLOBAL_BATCH_SIZE} / ${MICRO_BATCH_SIZE}"
echo "Seq length:  ${SEQ_LENGTH}"
echo "Model:       ${LOCAL_HF_PATH}"
if [[ ${BACKEND} == energon ]]; then
    echo "Dataset:     energon (${DATA_PATH})"
else
    echo "Dataset:     mock (${MOCK_NUM_BASE_EXAMPLES} synthetic examples)"
fi
echo "Save dir:    ${SAVE_DIR}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================================"

# ===================================== Execution =====================================
RECIPE_ARGS=(
    "${MEGATRON_PATH}/run_recipe.py"
    --recipe "${RECIPE}"
    --mode pretrain
)
if [[ ${MODE} == dryrun ]]; then
    RECIPE_ARGS+=(--dry-run)
fi
RECIPE_ARGS+=(
    "${DATA_CONFIG[@]}"
    "${TRAIN_CONFIG[@]}"
    "${PARALLEL_CONFIG[@]}"
    "${MODEL_CONFIG[@]}"
    "${CKPT_AND_LOGGING_CONFIG[@]}"
)
# Trailing KEY=VALUE arguments go last so they win over the blocks above.
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    RECIPE_ARGS+=("${EXTRA_ARGS[@]}")
fi

cd ${MEGATRON_PATH}/3rdparty/Megatron-Bridge
./scripts/switch_mcore.sh main
cd ${CURRENT_DIR}

case "${MODE}" in
    dryrun)
        python "${RECIPE_ARGS[@]}"
        echo "======================================"
        echo "Dry run complete. Resolved config: ${CONFIG_DUMP}"
        echo "======================================"
        ;;

    train)
        TORCHRUN_ARGS=(
            --nnodes=${NNODES}
            --nproc_per_node=${GPUS}
            --master_addr=${MASTER_ADDR}
            --master_port=${MASTER_PORT}
        )

        if (( NNODES == 1 )); then
            torchrun "${TORCHRUN_ARGS[@]}" "${RECIPE_ARGS[@]}" 2>&1 \
                | tee "${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log"
        else
            # One mpirun process per node (slots=1); each starts its own torchrun
            # for that node's GPUs. The remote command is assembled as a string
            # because --node_rank must stay literal until it is evaluated on the
            # remote host, where mpirun sets OMPI_COMM_WORLD_RANK.
            HOSTFILE_SLOTS="${CURRENT_DIR}/hostfile_slots"
            grep -ve '^[[:space:]]*$' ${HOSTFILE} | sed -n "1,${NNODES}p" | awk '{print $1" slots=1"}' > ${HOSTFILE_SLOTS}

            TORCHRUN_QUOTED="$(printf '%q ' "${TORCHRUN_ARGS[@]}")"
            RECIPE_QUOTED="$(printf '%q ' "${RECIPE_ARGS[@]}")"
            REMOTE_CMD="torchrun --node_rank=\${OMPI_COMM_WORLD_RANK} ${TORCHRUN_QUOTED} ${RECIPE_QUOTED}"

            # The remote shell re-sources env.sh, which rebuilds PYTHONPATH from
            # scratch, so the Energon checkout has to be re-appended there.
            REMOTE_PREFIX="source ${BRIDGE_ENV_SCRIPT}"
            if [[ -n ${EXTRA_PYTHONPATH} ]]; then
                REMOTE_PREFIX="${REMOTE_PREFIX} && export PYTHONPATH=\"\${PYTHONPATH}:${EXTRA_PYTHONPATH}\""
            fi

            # The recipe reads its model, data, batch, and mock knobs from the
            # environment, so every one of them has to cross to the other nodes.
            mpirun -np ${NNODES} --hostfile ${HOSTFILE_SLOTS} \
                --allow-run-as-root \
                --bind-to none \
                --mca plm_rsh_no_tree_spawn 1 \
                --mca plm_rsh_args "-p ${PORT}" \
                -x PATH \
                -x LIBRARY_PATH \
                -x LD_LIBRARY_PATH \
                -x PYTHONPATH \
                -x HYHAL_PATH \
                -x ROCM_PATH \
                -x PYTHONWARNINGS \
                -x TRANSFORMERS_VERBOSITY \
                -x LOCAL_HF_PATH \
                -x DATA_PATH \
                -x TRAIN_ITERS \
                -x GLOBAL_BATCH_SIZE \
                -x MICRO_BATCH_SIZE \
                -x MOCK_PROMPT \
                -x MOCK_NUM_IMAGES \
                -x MOCK_NUM_BASE_EXAMPLES \
                -x MOCK_RANDOM_SEED \
                -x MOCK_IMAGE_WIDTH \
                -x MOCK_IMAGE_HEIGHT \
                -x MASTER_ADDR=${MASTER_ADDR} \
                -x MASTER_PORT=${MASTER_PORT} \
                bash -c "
                ${REMOTE_PREFIX} && \
                ${REMOTE_CMD}" 2>&1 | tee "${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log"
        fi
        echo "======================================"
        echo "Training finished. Checkpoints: ${SAVE_DIR}"
        echo "======================================"
        ;;

    *)
        echo "ERROR: unknown mode '${MODE}'. Use 'dryrun' or 'train'."
        exit 1
        ;;
esac
