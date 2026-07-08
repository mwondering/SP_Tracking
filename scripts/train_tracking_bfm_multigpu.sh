#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LAUNCH_SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/train_tracking_bfm_multigpu.sh [motion_path] [hydra_overrides...]

Environment:
  SP_TRACKING_GPUS=0,1              CUDA devices visible to torchrun.
  SP_TRACKING_NPROC=2              Number of torchrun workers. Defaults to the GPU count.
  SP_TRACKING_MOTION_PATH=<path>   Motion directory when no positional path is given.

Examples:
  SP_TRACKING_GPUS=0,1 scripts/train_tracking_bfm_multigpu.sh
  SP_TRACKING_GPUS=0,1,2,3 scripts/train_tracking_bfm_multigpu.sh /path/to/motions task.num_envs=4096
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MOTION_PATH="${SP_TRACKING_MOTION_PATH:-/home/lenovo/DATASETS/Data10k_full}"
if [[ $# -gt 0 && "$1" != *=* ]]; then
  MOTION_PATH="$1"
  shift
fi

if [[ -z "${MOTION_PATH}" ]]; then
  usage
  exit 2
fi

GPUS="${SP_TRACKING_GPUS:-2,3}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NPROC="${SP_TRACKING_NPROC:-${#GPU_LIST[@]}}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cmd=(
  uv run torchrun
  --standalone
  "--nproc_per_node=${NPROC}"
  -m sp_tracking.scripts.train
  task=tracking_bfm
  "motion_path=${MOTION_PATH}"
  "launch_script_path=${LAUNCH_SCRIPT_PATH}"
  "task.num_envs=${SP_TRACKING_NUM_ENVS:-16}"
  "agent.max_iterations=${SP_TRACKING_MAX_ITERATIONS:-300000}"
  "agent.num_steps_per_env=${SP_TRACKING_NUM_STEPS_PER_ENV:-24}"
  "agent.logger=wandb"
  "agent.upload_model=False"
  "agent.wandb_project=${SP_TRACKING_WANDB_PROJECT:-sp-tracking}"
  "agent.save_interval=${SP_TRACKING_SAVE_INTERVAL:-1000}"
  "log_root=${SP_TRACKING_LOG_ROOT:-logs/rsl_rl}"
)

RUN_NAME="${SP_TRACKING_RUN_NAME:-trydebug_multigpu}"
if [[ -n "${RUN_NAME}" ]]; then
  cmd+=("agent.run_name=${RUN_NAME}")
fi

cd "${REPO_ROOT}"
exec "${cmd[@]}" "$@"
