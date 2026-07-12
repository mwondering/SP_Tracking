#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LAUNCH_SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/train_tracking_bfm.sh [motion_path] [hydra_overrides...]
  SP_TRACKING_MOTION_PATH=<motion_path> scripts/train_tracking_bfm.sh [hydra_overrides...]

Environment:
  SP_TRACKING_HISTORY_STEPS=0       Command history horizon; old h100 baseline default.
  SP_TRACKING_FUTURE_STEPS=1        Command future horizon; old h100 baseline default.

Examples:
  scripts/train_tracking_bfm.sh
  scripts/train_tracking_bfm.sh /path/to/motions
  scripts/train_tracking_bfm.sh /path/to/motions task.num_envs=2048 agent.max_iterations=50000
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

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cmd=(
  uv run sp-train
  task=tracking_bfm_sp
  "motion_path=${MOTION_PATH}"
  "launch_script_path=${LAUNCH_SCRIPT_PATH}"
  "task.num_envs=${SP_TRACKING_NUM_ENVS:-16}"
  "task.command.command.history_steps=${SP_TRACKING_HISTORY_STEPS:-0}"
  "task.command.command.future_steps=${SP_TRACKING_FUTURE_STEPS:-1}"
  "agent.max_iterations=${SP_TRACKING_MAX_ITERATIONS:-300000}"
  "agent.num_steps_per_env=${SP_TRACKING_NUM_STEPS_PER_ENV:-32}"
  "agent.logger=wandb"
  "agent.upload_model=False"
  "agent.wandb_project=${SP_TRACKING_WANDB_PROJECT:-sp-tracking}"
  "agent.save_interval=${SP_TRACKING_SAVE_INTERVAL:-1000}"
  "log_root=${SP_TRACKING_LOG_ROOT:-logs/rsl_rl}"
)

RUN_NAME="${SP_TRACKING_RUN_NAME:-trydebug}"
if [[ -n "${RUN_NAME}" ]]; then
  cmd+=("agent.run_name=${RUN_NAME}")
fi

cd "${REPO_ROOT}"
exec "${cmd[@]}" "$@"
