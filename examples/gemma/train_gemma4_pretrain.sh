#!/bin/bash

set -euo pipefail
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# default env
export LOCAL_HF_PATH=/path/to/gemma-4-E4B-it
export GEMMA4_CONVERSION_MODE=text
export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
# ===================================== Launch Mode =====================================
RECIPE=${1:-}
MODE=${2:-}
HOSTFILE=${3:-}
NODES=${4:-}
PROFILING="torch"

RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

# Discover supported recipes from the gemma4 module
SUPPORTED_RECIPES=(
    gemma4_e4b_pretrain_16gpu_bw1000_bf16_config
)

# Validate RECIPE argument
if [[ -z ${RECIPE} ]]; then
    echo -e "${RED}ERROR: missing recipe argument.${RESET}"
    echo -e "${YELLOW}Supported recipes:${RESET}"
    printf "  - %s\n" "${SUPPORTED_RECIPES[@]}"
    echo ""
    echo "Usage: $0 <RECIPE> <MODE> [HOSTFILE] [NODES]"
    echo "  MODE: dryrun | train"
    exit 1
fi

# ===================================== Multi-node Topology =====================================
GPUS="${GPUS:-8}"
PORT="${PORT:-11453}"
MASTER_ADDR=127.0.0.1
MASTER_PORT=29508

if [[ -n ${HOSTFILE} ]]; then
    [[ -f ${HOSTFILE} ]] || { echo "ERROR: ${HOSTFILE} not found."; exit 1; }
    HOSTFILE=$(cd $(dirname ${HOSTFILE}) && pwd)/$(basename ${HOSTFILE})
    AVAILABLE_NODES=$(grep -cve '^[[:space:]]*$' ${HOSTFILE})
    NNODES=${NODES:-${AVAILABLE_NODES}}
    if (( NNODES > AVAILABLE_NODES )); then
        echo "ERROR: node count ${NNODES} exceeds ${AVAILABLE_NODES} host(s) in ${HOSTFILE}."
        exit 1
    fi
    MASTER_ADDR=$(grep -ve '^[[:space:]]*$' ${HOSTFILE} | sed -n 1p | awk '{print $1}')
else
    NNODES=${NODES:-1}
    if (( NNODES > 1 )); then
        echo "ERROR: multi-node run requires a hostfile."
        exit 1
    fi
fi
WORLD_GPUS=$((NNODES * GPUS))

# ===================================== Data Config =====================================
SEQ_LENGTH=4096
DATASET_BLEND=null

DATA_CONFIG=(
    dataset.seq_length=${SEQ_LENGTH}
    dataset.blend=${DATASET_BLEND}
)

# ===================================== Train Config =====================================
TRAIN_ITERS=50
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=1

TRAIN_CONFIG=(
    train.train_iters=${TRAIN_ITERS}
    train.global_batch_size=${GLOBAL_BATCH_SIZE}
    train.micro_batch_size=${MICRO_BATCH_SIZE}
)

# ===================================== Optimizer Config =====================================
OPTIMIZER_CONFIG=(
    # optimizer.optimizer=adam
)

# ===================================== Parallel Config =====================================
TP="${TENSOR_PARALLEL:-4}"
PP="${PIPELINE_PARALLEL:-1}"
EP="${EXPERT_PARALLEL:-1}"
ETP="${EXPERT_TENSOR_PARALLEL:-1}"
CP="${CONTEXT_PARALLEL:-1}"

PARALLEL_CONFIG=(
    model.tensor_model_parallel_size=${TP}
    model.pipeline_model_parallel_size=${PP}
    model.expert_model_parallel_size=${EP}
    model.expert_tensor_parallel_size=${ETP}
    model.context_parallel_size=${CP}
)

# ===================================== Model Config =====================================
MODEL_CONFIG=(
    model.seq_length=${SEQ_LENGTH}
    ddp.overlap_grad_reduce=false
    # model.attention_backend=flash
    mixed_precision=bf16_mixed
    # scheduler.lr_warmup_iters=1
    optimizer.lr=2.0e-5
    optimizer.min_lr=2.0e-6

)

# ===================================== Checkpoint & Logging Config =====================================
SAVE_DIR=${CURRENT_DIR}/${RECIPE}_tp${TP}_pp${PP}_ep${EP}_cp${CP}
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
    profiling.profile_ranks=[0]
    profiling.profile_step_start=9
    profiling.profile_step_end=10
    profiling.pytorch_profiler_collect_callstack=true
)

# ===================================== Summary =====================================
mkdir -p "${SAVE_DIR}/train_logs"
echo "======================================"
echo -e "${YELLOW}Gemma 4 E4B pretrain${RESET}"
echo "======================================"
echo "Mode:        ${MODE}"
echo "Profiling:   ${PROFILING:-off}"
echo "Recipe:      ${RECIPE}"
echo "Conversion:  ${GEMMA4_CONVERSION_MODE}"
echo "Nodes:       ${NNODES}"
echo "GPUs:        ${GPUS}/node, $((NNODES * GPUS)) total   (TP=${TP}, PP=${PP}, EP=${EP})"
echo "Iters:       ${TRAIN_ITERS}"
echo "GBS/MBS:     ${GLOBAL_BATCH_SIZE} / ${MICRO_BATCH_SIZE}"
echo "Seq length:  ${SEQ_LENGTH}"
echo "Save dir:    ${SAVE_DIR}"
echo "Dataset:     mock (dataset.blend=${DATASET_BLEND})"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================================"

# ===================================== Execution =====================================
cd ${MEGATRON_PATH}/3rdparty/Megatron-Bridge
cd ${CURRENT_DIR}

env_args=(
    -x PATH
    -x LD_LIBRARY_PATH
    -x LOCAL_HF_PATH
    -x GPU_MAX_HW_QUEUES
    -x GEMMA4_CONVERSION_MODE
    -x NVTE_USE_HIPBLASLT_GROUPEDGEMM
    -x MASTER_ADDR=${MASTER_ADDR}
    -x MASTER_PORT=${MASTER_PORT}
)

case "${MODE}" in
    dryrun)
        source /opt/dtk/env.sh
        source ${MEGATRON_PATH}/env/env.sh
        python ${MEGATRON_PATH}/run_recipe.py \
            --recipe ${RECIPE} \
            --mode pretrain \
            --dry-run \
            "${DATA_CONFIG[@]}" \
            "${TRAIN_CONFIG[@]}" \
            "${OPTIMIZER_CONFIG[@]}" \
            "${PARALLEL_CONFIG[@]}" \
            "${MODEL_CONFIG[@]}" \
            "${CKPT_AND_LOGGING_CONFIG[@]}"
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
        RECIPE_ARGS=(
            ${MEGATRON_PATH}/run_recipe.py
            --recipe ${RECIPE}
            --mode pretrain
            ${DATA_CONFIG[@]}
            ${TRAIN_CONFIG[@]}
            ${OPTIMIZER_CONFIG[@]}
            ${PARALLEL_CONFIG[@]}
            ${MODEL_CONFIG[@]}
            ${CKPT_AND_LOGGING_CONFIG[@]}
        )

        if [[ $PROFILING == "torch" ]]; then
            RECIPE_ARGS+=(${PROFILING_CONFIG[@]})
        fi

        if (( NNODES == 1 )); then
            source /opt/dtk/env.sh
            source ${MEGATRON_PATH}/env/env.sh
            torchrun "${TORCHRUN_ARGS[@]}" "${RECIPE_ARGS[@]}" 2>&1 | tee "${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log"
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

            mpirun -np ${NNODES} --hostfile ${HOSTFILE_SLOTS} \
                --allow-run-as-root \
                --bind-to none \
                --mca plm_rsh_no_tree_spawn 1 \
                --mca plm_rsh_args "-p ${PORT}" \
                "${env_args[@]}" \
                bash -c "
                source /opt/dtk/env.sh && \
                source ${MEGATRON_PATH}/env/env.sh && \
                ${REMOTE_CMD}" 2>&1 | tee ${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log
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