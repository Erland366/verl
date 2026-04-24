#!/bin/bash

set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Activate your conda environment first, for example: conda activate verlrl" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.env"
    set +a
fi

TRAIN_FILES="${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}"
VAL_FILES="${VAL_FILES:-$HOME/data/gsm8k/test.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-verl_smoke}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_3b_4gpu_vllm_smoke}"
WANDB_ENTITY="${WANDB_ENTITY:-erlandpg}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-32}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-4}"
ROLLOUT_DP_EXTERNAL_LB="${ROLLOUT_DP_EXTERNAL_LB:-True}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-False}"
VLLM_COMPILATION_MODE="${VLLM_COMPILATION_MODE:-0}"
VLLM_COMPILATION_BACKEND="${VLLM_COMPILATION_BACKEND:-eager}"
VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE:-NONE}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-/tmp/vllm-v0.10.0}"
AMDSMI_PYTHONPATH="${AMDSMI_PYTHONPATH:-/opt/rocm-6.3.3/share/amd_smi}"
HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
RCCL_MSCCL_ENABLE="${RCCL_MSCCL_ENABLE:-0}"
RCCL_MSCCLPP_ENABLE="${RCCL_MSCCLPP_ENABLE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "${TRAIN_FILES}" ]; then
    echo "Missing train parquet: ${TRAIN_FILES}" >&2
    exit 1
fi

if [ ! -f "${VAL_FILES}" ]; then
    echo "Missing validation parquet: ${VAL_FILES}" >&2
    exit 1
fi

export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export WANDB_ENTITY
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-1}"
export HSA_FORCE_FINE_GRAIN_PCIE
export RCCL_MSCCL_ENABLE
export RCCL_MSCCLPP_ENABLE

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
fi

if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3
fi
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

if [ "${ROLLOUT_NAME}" = "vllm" ]; then
    if [ ! -d "${VLLM_SOURCE_ROOT}/vllm" ]; then
        echo "Missing VLLM_SOURCE_ROOT checkout: ${VLLM_SOURCE_ROOT}" >&2
        exit 1
    fi

    if [ ! -e "${VLLM_SOURCE_ROOT}/vllm/_C.abi3.so" ]; then
        echo "Missing vLLM core extension in ${VLLM_SOURCE_ROOT}/vllm" >&2
        exit 1
    fi

    export PYTHONPATH="${AMDSMI_PYTHONPATH}:${VLLM_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

CMD=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator="${ADV_ESTIMATOR}"
    trainer.val_before_train=False
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_max_samples="${TRAIN_MAX_SAMPLES}"
    data.val_max_samples="${VAL_MAX_SAMPLES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.dataloader_num_workers=0
    actor_rollout_ref.model.path="${MODEL_PATH}"
    +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}"
    actor_rollout_ref.actor.fsdp_config.use_torch_compile="${USE_TORCH_COMPILE}"
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}"
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}"
    actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP_SIZE}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
    +actor_rollout_ref.rollout.enable_sleep_mode="${ROLLOUT_ENABLE_SLEEP_MODE}"
    actor_rollout_ref.rollout.top_k=0
    actor_rollout_ref.rollout.do_sample=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024
    +actor_rollout_ref.rollout.engine_kwargs.vllm.data_parallel_external_lb="${ROLLOUT_DP_EXTERNAL_LB}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode="${VLLM_COMPILATION_MODE}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.backend="${VLLM_COMPILATION_BACKEND}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="${VLLM_CUDAGRAPH_MODE}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.use_torch_compile="${USE_TORCH_COMPILE}"
    actor_rollout_ref.ref.fsdp_config.use_torch_compile="${USE_TORCH_COMPILE}"
    critic.optim.lr=1e-5
    critic.model.path="${MODEL_PATH}"
    +critic.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}"
    critic.model.use_remove_padding="${USE_REMOVE_PADDING}"
    critic.model.enable_gradient_checkpointing=True
    critic.ppo_mini_batch_size=8
    critic.ppo_micro_batch_size_per_gpu=1
    critic.ppo_epochs=1
    critic.fsdp.use_torch_compile="${USE_TORCH_COMPILE}"
    critic.ppo_max_token_len_per_gpu=1024
    critic.forward_max_token_len_per_gpu=1024
    critic.fsdp.param_offload=False
    critic.fsdp.optimizer_offload=False
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.use_legacy_worker_impl=disable
    +ray_kwargs.ray_init.address=local
)

echo "Running 4-GPU smoke test with:"
echo "  CONDA_PREFIX=${CONDA_PREFIX}"
echo "  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  ADV_ESTIMATOR=${ADV_ESTIMATOR}"
echo "  ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "  TRAIN_FILES=${TRAIN_FILES}"
echo "  VAL_FILES=${VAL_FILES}"
echo "  TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES}"
echo "  VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}"
echo "  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}"
echo "  MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "  TOTAL_EPOCHS=${TOTAL_EPOCHS}"
echo "  ROLLOUT_NAME=${ROLLOUT_NAME}"
echo "  ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE}"
echo "  ROLLOUT_DP_SIZE=${ROLLOUT_DP_SIZE}"
echo "  ROLLOUT_DP_EXTERNAL_LB=${ROLLOUT_DP_EXTERNAL_LB}"
echo "  ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "  ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER}"
echo "  ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE}"
echo "  ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE}"
echo "  VLLM_COMPILATION_MODE=${VLLM_COMPILATION_MODE}"
echo "  VLLM_COMPILATION_BACKEND=${VLLM_COMPILATION_BACKEND}"
echo "  VLLM_CUDAGRAPH_MODE=${VLLM_CUDAGRAPH_MODE}"
echo "  USE_TORCH_COMPILE=${USE_TORCH_COMPILE}"
echo "  USE_REMOVE_PADDING=${USE_REMOVE_PADDING}"
echo "  VLLM_SOURCE_ROOT=${VLLM_SOURCE_ROOT}"
echo "  AMDSMI_PYTHONPATH=${AMDSMI_PYTHONPATH}"
echo "  VLLM_USE_V1=${VLLM_USE_V1}"
echo "  RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES-<unset>}"
echo "  HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE}"
echo "  RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE}"
echo "  RCCL_MSCCLPP_ENABLE=${RCCL_MSCCLPP_ENABLE}"
echo "  WANDB_ENTITY=${WANDB_ENTITY}"
echo "  PYTHONPATH=${PYTHONPATH-<unset>}"

if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
