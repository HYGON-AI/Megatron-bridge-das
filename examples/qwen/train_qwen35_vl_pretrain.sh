#!/bin/bash
# ==============================================================================
# Qwen3.5-VL pretrain smoke test (mpirun + torchrun), BW1100 / BW1000
#
# Recipe names carry both the model size and the platform, e.g.
#   qwen35_vl_9b_pretrain_8gpu_bw1100_bf16_config       (Energon dataset)
#   qwen35_vl_9b_pretrain_8gpu_bw1100_bf16_mock_config  (synthetic data)
# ==============================================================================

set -euo pipefail
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# default env
# PORT is the ssh port mpirun uses to reach the other nodes, so it has to match
# the container's sshd port (site.json container.ssh_port), not the default 22.
PORT="${PORT:-25951}"
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/Qwen3.5-VL-2B}
BRIDGE_ENV_SCRIPT="${BRIDGE_ENV_SCRIPT:-${MEGATRON_PATH}/env/env.sh}" # cluster env script (NCCL/IB vars, PYTHONPATH, HCU recipe registration); override with --env BRIDGE_ENV_SCRIPT=<path> to use a different cluster's env.sh

# env.sh does not cover Megatron-Energon; add it when a local checkout is
# present. Remembered separately because the remote shell of a multi-node run
# re-sources env.sh and would otherwise lose it.
ENERGON_SRC="${MEGATRON_PATH}/3rdparty/Megatron-Energon/src"
EXTRA_PYTHONPATH=""
if [[ -d "${ENERGON_SRC}" ]]; then
    EXTRA_PYTHONPATH="${ENERGON_SRC}"
fi

# ===================================== Launch Mode =====================================
RECIPE=${1:-}
MODE=${2:-}
shift 2 2>/dev/null || true

# [HOSTFILE] and [NODES] keep deepseek's positions, and anything left over is a
# KEY=VALUE override forwarded to the runner. A third argument containing '='
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

PROFILING="" # set to "torch" to capture a PyTorch profile on ranks 0 and 4

RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

# Recipes published by hcu_megatron.bridge.recipes.qwen.bw1100 / .bw1000.
# Each size has an Energon variant and a synthetic-data (mock) variant.
SUPPORTED_RECIPES=(
    qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_config
    qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_mock_config
    qwen35_vl_4b_pretrain_8gpu_bw1100_bf16_config
    qwen35_vl_4b_pretrain_8gpu_bw1100_bf16_mock_config
    qwen35_vl_9b_pretrain_8gpu_bw1100_bf16_config
    qwen35_vl_9b_pretrain_8gpu_bw1100_bf16_mock_config
    qwen35_vl_27b_pretrain_8gpu_bw1100_bf16_config
    qwen35_vl_27b_pretrain_8gpu_bw1100_bf16_mock_config
    qwen35_vl_35b_a3b_pretrain_16gpu_bw1100_bf16_config
    qwen35_vl_35b_a3b_pretrain_16gpu_bw1100_bf16_mock_config
    qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config
    qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_mock_config
    qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_mock_config
    qwen35_vl_27b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_27b_pretrain_8gpu_bw1000_bf16_mock_config
    qwen35_vl_35b_a3b_pretrain_16gpu_bw1000_bf16_config
    qwen35_vl_35b_a3b_pretrain_16gpu_bw1000_bf16_mock_config
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

# Size, platform and dataset backend are carried by the recipe name itself:
#   qwen35_vl_<size-slug>_pretrain_<N>gpu_<platform>_bf16[_mock]_config
# The slug is lower-case and may carry an MoE suffix (35b_a3b), so it is
# matched up to the "_pretrain_" separator rather than a bare "<digits>b"
# pattern.
SIZE=$(echo "${RECIPE}" | sed -n 's/^qwen35_vl_\([0-9][0-9a-z_]*[0-9a-z]\)_pretrain_.*/\1/p')
if [[ -z ${SIZE} ]]; then
    echo -e "${RED}ERROR: cannot parse model size from recipe '${RECIPE}'.${RESET}"
    exit 1
fi
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

# One mpirun process per node (slots=1); each starts its own torchrun for that
# node's GPUs. Only the multi-node branch uses this file; single-node runs have
# no hostfile at all, and feeding grep an empty HOSTFILE would make it exit 1
# and kill the script under `set -euo pipefail`.
HOSTFILE_SLOTS="${CURRENT_DIR}/hostfile_slots"
if [[ -n ${HOSTFILE} ]]; then
    grep -ve '^[[:space:]]*$' "${HOSTFILE}" | sed -n "1,${NNODES}p" | awk '{print $1" slots=1"}' > "${HOSTFILE_SLOTS}"
fi

# ===================================== Model and Data Paths =====================================
# The recipe name picks the topology and LOCAL_HF_PATH picks the weights, so
# the two have to agree. Only existence is checked here; the architecture
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
    SEQ_LENGTH=${SEQ_LENGTH:-2048}
else
    SEQ_LENGTH=${SEQ_LENGTH:-4096}
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

# ===================================== Optimizer Config =====================================
OPTIMIZER_CONFIG=(
    # optimizer.optimizer=adam
)

# ===================================== Parallel Config =====================================
# Defaults mirror the recipe's own topology (qwen35_vl.py's spec table), which
# is identical for bw1000 and bw1100. 
case "${SIZE}" in
    2b)        TP_DEFAULT=1; PP_DEFAULT=1; SP_DEFAULT=false; EP_DEFAULT=1 ;;
    4b)        TP_DEFAULT=1; PP_DEFAULT=1; SP_DEFAULT=false;  EP_DEFAULT=1 ;;
    9b)        TP_DEFAULT=1; PP_DEFAULT=1; SP_DEFAULT=false;  EP_DEFAULT=1 ;;
    27b)       TP_DEFAULT=2; PP_DEFAULT=2; SP_DEFAULT=true;  EP_DEFAULT=1 ;;
    # SP requires TP>1, so the TP1 MoE topology keeps it off.
    35b_a3b)   TP_DEFAULT=1; PP_DEFAULT=2; SP_DEFAULT=false; EP_DEFAULT=8 ;;
    *)  echo -e "${RED}ERROR: unknown model size '${SIZE}' in ${RECIPE}.${RESET}"; exit 1 ;;
esac
TP=${TENSOR_PARALLEL:-${TP_DEFAULT}}
PP=${PIPELINE_PARALLEL:-${PP_DEFAULT}}
CP=${CONTEXT_PARALLEL:-1}
EP=${EXPERT_PARALLEL:-${EP_DEFAULT}}
ETP=${EXPERT_TENSOR_PARALLEL:-1}
SP=${SEQUENCE_PARALLEL:-${SP_DEFAULT}}

PARALLEL_CONFIG=(
    model.tensor_model_parallel_size=${TP}
    model.pipeline_model_parallel_size=${PP}
    model.context_parallel_size=${CP}
)
if (( EP > 1 )); then
    PARALLEL_CONFIG+=(
        model.expert_model_parallel_size="${EP}"
        model.expert_tensor_parallel_size="${ETP}"
    )
fi

# Catch an unusable topology here rather than several minutes into startup.
# MoE sizes additionally require the expert grid (ETP*EP*PP) to divide the
# world size, exactly like Megatron's own parallel-layout check.
MODEL_PARALLEL=$((TP * PP * CP))
EXPERT_PARALLEL=$((ETP * EP * PP))
if (( EP > 1 && WORLD_GPUS % EXPERT_PARALLEL != 0 )); then
    echo -e "${RED}ERROR: ${WORLD_GPUS} GPUs is not divisible by ETP*EP*PP=${EXPERT_PARALLEL}.${RESET}"
    exit 1
fi
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
# These are performance runs: never read or write checkpoints, so nothing is
# left behind on disk and a stale checkpoint in SAVE_DIR can't be picked up.
SAVE_DIR=${CHECKPOINT_PATH:-${CURRENT_DIR}/${RECIPE}_tp${TP}_pp${PP}_cp${CP}}
CONFIG_DUMP=${SAVE_DIR}/${RECIPE}.yaml

CKPT_AND_LOGGING_CONFIG=(
    checkpoint.load=null
    checkpoint.save=null
    checkpoint.save_interval=100
    logger.log_interval=1
    logger.logging_level=40
    logger.log_throughput=true
    logger.tensorboard_dir=${SAVE_DIR}/tb_logs
    logger.save_config_filepath=${CONFIG_DUMP}
)

# ===================================== Profiler Config =====================================
PROFILING_CONFIG=(
    profiling.use_pytorch_profiler=true
    profiling.profile_ranks=[0,4]
    profiling.profile_step_start=10
    profiling.profile_step_end=11
    profiling.pytorch_profiler_collect_callstack=true
)

# ===================================== Summary =====================================
mkdir -p ${SAVE_DIR}/train_logs
echo "======================================"
echo -e "${YELLOW}Qwen3.5-VL ${SIZE_UPPER} pretrain${RESET}"
echo "======================================"
echo "Mode:        ${MODE}"
echo "Profiling:   ${PROFILING:-off}"
echo "Recipe:      ${RECIPE}"
echo "Nodes:       ${NNODES}${HOSTFILE:+ (${HOSTFILE})}"
echo "GPUs:        ${GPUS}/node, ${WORLD_GPUS} total   (TP=${TP}, PP=${PP}, CP=${CP}, EP=${EP}, SP=${SP}, DP=${DP})"
echo "Iters:       ${TRAIN_ITERS}"
echo "GBS/MBS:     ${GLOBAL_BATCH_SIZE} / ${MICRO_BATCH_SIZE}"
echo "Seq length:  ${SEQ_LENGTH}"
echo "Model:       ${LOCAL_HF_PATH}"
if [[ ${BACKEND} == energon ]]; then
    echo "Dataset:     energon (${DATA_PATH})"
else
    echo "Dataset:     mock (${MOCK_NUM_BASE_EXAMPLES} synthetic examples)"
fi
echo "Save dir:    ${SAVE_DIR}   (no checkpoint load/save)"
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
    "${OPTIMIZER_CONFIG[@]}"
    "${PARALLEL_CONFIG[@]}"
    "${MODEL_CONFIG[@]}"
    "${CKPT_AND_LOGGING_CONFIG[@]}"
)
if [[ ${PROFILING} == "torch" ]]; then
    RECIPE_ARGS+=("${PROFILING_CONFIG[@]}")
fi
# Trailing KEY=VALUE arguments go last so they win over the blocks above.
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    RECIPE_ARGS+=("${EXTRA_ARGS[@]}")
fi

cd ${MEGATRON_PATH}/3rdparty/Megatron-Bridge
./scripts/switch_mcore.sh main
cd ${CURRENT_DIR}

# Every brace runs /opt/dtk/env.sh before the cluster env script: without it
# torch sees zero HCUs and dies with "RuntimeError: No CUDA GPUs are available"
# (this bit the single-node branch of train_qwen35_text_pretrain.sh before it
# was fixed -- see QA.md).
env_args=(
    -x PATH
    -x LIBRARY_PATH
    -x LD_LIBRARY_PATH
    -x PYTHONPATH
    -x HYHAL_PATH
    -x ROCM_PATH
    -x PYTHONWARNINGS
    -x TRANSFORMERS_VERBOSITY
    -x LOCAL_HF_PATH
    -x DATA_PATH
    -x TRAIN_ITERS
    -x GLOBAL_BATCH_SIZE
    -x MICRO_BATCH_SIZE
    -x MOCK_PROMPT
    -x MOCK_NUM_IMAGES
    -x MOCK_NUM_BASE_EXAMPLES
    -x MOCK_RANDOM_SEED
    -x MOCK_IMAGE_WIDTH
    -x MOCK_IMAGE_HEIGHT
    -x MASTER_ADDR="${MASTER_ADDR}"
    -x MASTER_PORT="${MASTER_PORT}"
)
SAVE_LOG=${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log

case "${MODE}" in
    dryrun)
        # shellcheck source=/dev/null
        source /opt/dtk/env.sh
        # shellcheck source=/dev/null
        source "${BRIDGE_ENV_SCRIPT}"
        if [[ -n ${EXTRA_PYTHONPATH} ]]; then
            export PYTHONPATH="${PYTHONPATH}:${EXTRA_PYTHONPATH}"
        fi
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
            # shellcheck source=/dev/null
            source /opt/dtk/env.sh
            # shellcheck source=/dev/null
            source "${BRIDGE_ENV_SCRIPT}"
            if [[ -n ${EXTRA_PYTHONPATH} ]]; then
                export PYTHONPATH="${PYTHONPATH}:${EXTRA_PYTHONPATH}"
            fi
            torchrun "${TORCHRUN_ARGS[@]}" "${RECIPE_ARGS[@]}" 2>&1 | tee "${SAVE_LOG}"
        else
            # One mpirun process per node (slots=1); each starts its own torchrun
            # for that node's GPUs. The remote command is assembled as a string
            # because --node_rank must stay literal until it is evaluated on the
            # remote host, where mpirun sets OMPI_COMM_WORLD_RANK.
            TORCHRUN_QUOTED="$(printf '%q ' "${TORCHRUN_ARGS[@]}")"
            RECIPE_QUOTED="$(printf '%q ' "${RECIPE_ARGS[@]}")"
            REMOTE_CMD="torchrun --node_rank=\${OMPI_COMM_WORLD_RANK} ${TORCHRUN_QUOTED} ${RECIPE_QUOTED}"

            # The remote shell re-sources env.sh, which rebuilds PYTHONPATH from
            # scratch, so the Energon checkout has to be re-appended there.
            REMOTE_PREFIX="source /opt/dtk/env.sh && source ${BRIDGE_ENV_SCRIPT}"
            if [[ -n ${EXTRA_PYTHONPATH} ]]; then
                REMOTE_PREFIX="${REMOTE_PREFIX} && export PYTHONPATH=\"\${PYTHONPATH}:${EXTRA_PYTHONPATH}\""
            fi

            mpirun -np ${NNODES} --hostfile ${HOSTFILE_SLOTS} \
                --allow-run-as-root \
                --bind-to none \
                --mca plm_rsh_no_tree_spawn 1 \
                --mca plm_rsh_args "-p ${PORT}" \
                "${env_args[@]}" \
                bash -c "
                ${REMOTE_PREFIX} && \
                ${REMOTE_CMD}" 2>&1 | tee "${SAVE_LOG}"
        fi
        echo "======================================"
        echo "Training finished. Checkpoints: ${SAVE_DIR}"
        echo "======================================"
        ;;

    *)
        echo -e "${RED}ERROR: unknown mode '${MODE}'. Use 'dryrun' or 'train'.${RESET}" >&2
        exit 1
        ;;
esac
