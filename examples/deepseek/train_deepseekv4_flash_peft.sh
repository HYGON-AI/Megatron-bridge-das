#!/bin/bash

set -euo pipefail
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
BRIDGE_ENV_SCRIPT="${BRIDGE_ENV_SCRIPT:-${MEGATRON_PATH}/env/env.sh}" # cluster env script (NCCL/IB vars, PYTHONPATH, HCU recipe registration); override with --env BRIDGE_ENV_SCRIPT=<path>

# default env
PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=${DATA_PATH:-/path/to/Chinese-DeepSeek-R1-Distill-data-110k-SFT}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/DeepSeek-V4-Flash-0731}
export LOCAL_CORE_PATH=${LOCAL_CORE_PATH:-/path/to/DeepSeek-V4-Flash-0731-core}
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1

# ===================================== Launch Mode =====================================
RECIPE=${1:-}
MODE=${2:-}
HOSTFILE=${3:-}
NODES=${4:-}
PROFILING="" # If you want to profiling, please set this var to torch

RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

# Discover supported recipes from the deepseek module
SUPPORTED_RECIPES=(
    deepseek_v4_flash_peft_32gpu_bw1101_bf16_config
)

# Validate RECIPE argument
if [[ -z ${RECIPE} ]]; then
    echo -e "${RED}ERROR: missing recipe argument.${RESET}"
    echo -e "${YELLOW}Supported recipes:${RESET}"
    printf "  - %s\n" "${SUPPORTED_RECIPES[@]}"
    echo ""
    echo "Usage: $0 <RECIPE> <MODE> [HOSTFILE] [NODES]"
    echo "  MODE: dryrun | peft"
    exit 1
fi

# ===================================== Multi-node Topology =====================================
if [[ ${RECIPE} == *bw1300* ]]; then
    GPUS=${GPUS:-4}
else
    GPUS=${GPUS:-8}
fi
MASTER_ADDR=127.0.0.1
MASTER_PORT=29501

# NNODES comes from the hostfile line count, or $3, or 1.
if [[ -n ${HOSTFILE} ]]; then
    [[ -f ${HOSTFILE} ]] || { echo "ERROR: ${HOSTFILE} not found."; exit 1; }
    HOSTFILE=$(cd $(dirname ${HOSTFILE}) && pwd)/$(basename ${HOSTFILE})
    AVAILABLE_NODES=$(grep -cve '^[[:space:]]*$' ${HOSTFILE})
    NNODES=${NODES:-${AVAILABLE_NODES}}
    if (( NNODES > AVAILABLE_NODES )); then
        echo "ERROR: node count ${NNODES} exceeds ${AVAILABLE_NODES} host(s) in ${HOSTFILE}."
        exit 1
    fi
    # First host is the rendezvous master; mpirun places local rank 0 there.
    MASTER_ADDR=$(grep -ve '^[[:space:]]*$' ${HOSTFILE} | sed -n 1p | awk '{print $1}')
else
    NNODES=${NODES:-1}
    if (( NNODES > 1 )); then
        echo "ERROR: multi-node run requires a hostfile."
        exit 1
    fi
fi

# One mpirun process per node (slots=1); each starts its own torchrun
# for that node's GPUs. The remote command is assembled as a string
# because --node_rank must stay literal until it is evaluated on the
# remote host, where mpirun sets OMPI_COMM_WORLD_RANK.
HOSTFILE_SLOTS=${CURRENT_DIR}/hostfile_slots
grep -ve '^[[:space:]]*$' ${HOSTFILE} | sed -n "1,${NNODES}p" | awk '{print $1" slots=1"}' > ${HOSTFILE_SLOTS}
WORLD_GPUS=$((NNODES * GPUS))

# ===================================== Data Config =====================================
SEQ_LENGTH=4096

DATA_CONFIG=(
    dataset.dataset_root=${DATA_PATH}
    dataset.seq_length=${SEQ_LENGTH}
    dataset.do_test=false
)

# ===================================== Train Config =====================================
TRAIN_ITERS=1000
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1

TRAIN_CONFIG=(
    train.train_iters=${TRAIN_ITERS}
    train.global_batch_size=${GLOBAL_BATCH_SIZE}
    train.micro_batch_size=${MICRO_BATCH_SIZE}
    validation.eval_iters=-1
    ddp.check_for_nan_in_grad=false
    scheduler.lr_warmup_iters=100
    scheduler.lr_decay_iters=1000
    tokenizer.tokenizer_model=${LOCAL_HF_PATH}
)

# ===================================== Optimizer Config =====================================
OPTIMIZER_CONFIG=(
    optimizer.lr=1.0e-4
)

# ===================================== Parallel Config =====================================
TP=1
PP=4
EP=8
ETP=1
CP=1

PARALLEL_CONFIG=(
    model.tensor_model_parallel_size=${TP}
    model.pipeline_model_parallel_size=${PP}
    model.expert_model_parallel_size=${EP}
    model.expert_tensor_parallel_size=${ETP}
    model.context_parallel_size=${CP}
    model.pipeline_model_parallel_layout="Et*10|t*12|t*11|t*10mL"
)

# ===================================== Model Config =====================================
MODEL_CONFIG=(
    model.seq_length=${SEQ_LENGTH}
    model.sequence_parallel=true
    model.overlap_p2p_comm=true
    model.cross_entropy_loss_fusion=true
    model.cross_entropy_fusion_impl=te
    model.moe_router_fusion=true
    model.moe_permute_fusion=true
    model.moe_router_load_balancing_type=seq_aux_loss
    model.moe_router_pre_softmax=true
    model.moe_token_dispatcher_type=flex
    model.moe_flex_dispatcher_backend=deepep
    model.moe_flex_dispatcher_num_sms=60
    model.apply_dsa_kernel_fusion=null
    model.dsa_kernel_backend=hip
    model.use_fused_mhc=true
)

# ===================================== Checkpoint & Logging Config =====================================
SAVE_DIR=${CURRENT_DIR}/${RECIPE}_tp${TP}_pp${PP}_ep${EP}_cp${CP}
CONFIG_DUMP=${SAVE_DIR}/${RECIPE}.yaml

CKPT_AND_LOGGING_CONFIG=(
    checkpoint.pretrained_checkpoint=${LOCAL_CORE_PATH}/iter_0000000
    checkpoint.load=${SAVE_DIR}/ckpt
    checkpoint.load_optim=false
    checkpoint.load_rng=false
    checkpoint.save=${SAVE_DIR}/ckpt
    checkpoint.save_optim=false
    checkpoint.save_interval=300
    logger.log_interval=1
    logger.logging_level=40
    logger.log_throughput=true
    logger.tensorboard_dir=${SAVE_DIR}/tb_logs
    logger.save_config_filepath=${CONFIG_DUMP}
)

# ===================================== Profiler Config =====================================
PROFILING_CONFIG=(
    profiling.use_pytorch_profiler=true
    profiling.profile_ranks=[0,8,16,24]
    profiling.profile_step_start=3
    profiling.profile_step_end=4
    profiling.pytorch_profiler_collect_callstack=true
)

# ===================================== Summary =====================================
mkdir -p ${SAVE_DIR}/peft_logs
echo "======================================"
echo -e "${YELLOW}DeepSeek-V4-Flash peft${RESET}"
echo "======================================"
echo "Mode:        ${MODE}"
echo "Recipe:      ${RECIPE}"
echo "Nodes:       ${NNODES}"
echo "GPUs:        ${GPUS}/node, ${WORLD_GPUS} total   (TP=${TP}, PP=${PP}, EP=${EP})"
echo "Iters:       ${TRAIN_ITERS}"
echo "GBS/MBS:     ${GLOBAL_BATCH_SIZE} / ${MICRO_BATCH_SIZE}"
echo "Seq length:  ${SEQ_LENGTH}"
echo "Save dir:    ${SAVE_DIR}"
echo "Dataset:     ${DATA_PATH}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================================"

# ===================================== Execution =====================================
# deepseekv4 flash need switch Megatron-LM to dev branch before running
cd ${MEGATRON_PATH}/3rdparty/Megatron-Bridge
./scripts/switch_mcore.sh dev
cd ${CURRENT_DIR}

env_args=(
    -x PATH
    -x LD_LIBRARY_PATH
    -x LOCAL_HF_PATH
    -x NVTE_USE_HIPBLASLT_GROUPEDGEMM
    -x MASTER_ADDR=${MASTER_ADDR}
    -x MASTER_PORT=${MASTER_PORT}
)
SAVE_LOG=${SAVE_DIR}/peft_logs/mpi_$(date +%F-%H%M).log

# ------------------------------------------------------------------------------
# Phase 1: HF -> Megatron import (skipped if the base checkpoint already exists)
# ------------------------------------------------------------------------------

if [ -e "${LOCAL_CORE_PATH}/iter_0000000/.metadata" ]; then
    echo -e "${YELLOW}Phase 1: base checkpoint already present at ${LOCAL_CORE_PATH} - skipping import.${RESET}"
else
    echo -e "${YELLOW}Phase 1: importing ${LOCAL_HF_PATH} -> ${LOCAL_CORE_PATH} (TP=$TP PP=$PP EP=$EP)${RESET}"
    mpirun -np ${NNODES} \
           --hostfile ${HOSTFILE_SLOTS} \
           --allow-run-as-root \
           --bind-to none \
           --mca plm_rsh_no_tree_spawn 1 \
           --mca plm_rsh_args "-p ${PORT}" \
           "${env_args[@]}" \
           bash -c "
           source /opt/dtk/env.sh && \
           source ${BRIDGE_ENV_SCRIPT} && \
           cd ${MEGATRON_PATH}/scripts/conversion && \
           bash convert.sh import \
           --executor manual \
           --device gpu \
           --nodes ${NNODES} \
           --gpus-per-node ${GPUS} \
           --node-rank \${OMPI_COMM_WORLD_RANK} \
           --master-addr ${MASTER_ADDR} \
           --master-port ${MASTER_PORT} \
           --hf-model ${LOCAL_HF_PATH} \
           --megatron-path ${LOCAL_CORE_PATH} \
           --torch-dtype bfloat16 \
           --trust-remote-code \
           --moe-flex-dispatcher-backend deepep \
           --moe-flex-dispatcher-num-sms 60 \
           --tp $TP --pp $PP --ep $EP" 2>&1 | tee "${SAVE_LOG}"
fi

# ------------------------------------------------------------------------------
# Phase 2: Full PEFT
#
# No --dataset: the recipe ships an unpacked (SBHD) SQuAD config. To use another
# source, pass `--dataset gsm8k` or `--dataset local-jsonl dataset.dataset_root=<path>`.
# Do not enable `dataset.enable_offline_packing`; DSv4 rejects packed sequences.
# ------------------------------------------------------------------------------

echo -e "${YELLOW}Phase 2: PEFT with recipe ${RECIPE}${RESET}"
case "${MODE}" in
    dryrun)
        source ${BRIDGE_ENV_SCRIPT}
        python ${MEGATRON_PATH}/run_recipe.py \
            --recipe ${RECIPE} \
            --mode lora \
            --step-func dsv4_step \
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

    peft)
        TORCHRUN_ARGS=(
            --nnodes=${NNODES}
            --nproc_per_node=${GPUS}
            --master_addr=${MASTER_ADDR}
            --master_port=${MASTER_PORT}
        )
        RECIPE_ARGS=(
            ${MEGATRON_PATH}/run_recipe.py
            --recipe ${RECIPE}
            --mode lora
            --step-func dsv4_step
            --dataset local-jsonl
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
            source ${BRIDGE_ENV_SCRIPT}
            torchrun "${TORCHRUN_ARGS[@]}" "${RECIPE_ARGS[@]}" 2>&1 | tee "${SAVE_LOG}"
        else
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
                source ${BRIDGE_ENV_SCRIPT} && \
                ${REMOTE_CMD}" 2>&1 | tee "${SAVE_LOG}"
        fi
        echo "======================================"
        echo "Training finished. Checkpoints: ${SAVE_DIR}"
        echo "======================================"
        ;;

    *)
        echo "ERROR: unknown mode '${MODE}'. Use 'dryrun' or 'peft'."
        exit 1
        ;;
esac
