#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LAUNCH_SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/train_tracking_bfm_multigpu.sh [task_id] [motion_path] [hydra_overrides...]

Environment:
  SP_TRACKING_TASK_ID=<id>          Public task ID passed to the training entrypoint.
  SP_TRACKING_GPUS=0,1              CUDA devices visible to torchrun.
  SP_TRACKING_NPROC=2              Number of torchrun workers. Defaults to the GPU count.
  SP_TRACKING_MOTION_PATH=<path>   Motion directory when no positional path is given.
  SP_TRACKING_HISTORY_STEPS=0      Command history horizon; old h100 baseline default.
  SP_TRACKING_FUTURE_STEPS=1       Command future horizon; old h100 baseline default.

Examples:
  SP_TRACKING_GPUS=0,1 scripts/train_tracking_bfm_multigpu.sh
  scripts/train_tracking_bfm_multigpu.sh \
    SPTracking-G1-BFM-WBTeleopActor-HEFTCritic-HEFTReward /path/to/motions
  SP_TRACKING_GPUS=0,1,2,3 scripts/train_tracking_bfm_multigpu.sh /path/to/motions task.num_envs=4096
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASK_ID="${SP_TRACKING_TASK_ID:-SPTracking-G1-BFM-BFMActor-BFMCritic}"
if [[ "${1:-}" == SPTracking-* ]]; then
  TASK_ID="$1"
  shift
fi

MOTION_PATH="${SP_TRACKING_MOTION_PATH:-/data_zcy/zcy/datasets/motion_data_used_g1/AMASS_LAFAN_Qingtong/lafan_qingtong}"
if [[ $# -gt 0 && "$1" != *=* ]]; then
  MOTION_PATH="$1"
  shift
fi

if [[ -z "${MOTION_PATH}" ]]; then
  usage
  exit 2
fi

GPUS="${SP_TRACKING_GPUS:-0,1}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NPROC="${SP_TRACKING_NPROC:-${#GPU_LIST[@]}}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

HAS_TASK_OVERRIDE=false
for argument in "$@"; do
  if [[ "${argument}" == task=* || "${argument}" == task_id=* ]]; then
    HAS_TASK_OVERRIDE=true
    break
  fi
done

cmd=(
  uv run torchrun
  --standalone
  # "--local_ranks_filter=0"
  "--nproc_per_node=${NPROC}"
  -m sp_tracking.scripts.train
  "motion_path=${MOTION_PATH}"
  "launch_script_path=${LAUNCH_SCRIPT_PATH}"
  "task.num_envs=${SP_TRACKING_NUM_ENVS:-16384}"
  "task.command.command.history_steps=${SP_TRACKING_HISTORY_STEPS:-0}"
  "task.command.command.future_steps=${SP_TRACKING_FUTURE_STEPS:-1}"
  "agent.max_iterations=${SP_TRACKING_MAX_ITERATIONS:-300000}"
  "agent.num_steps_per_env=${SP_TRACKING_NUM_STEPS_PER_ENV:-32}"
  "agent.logger=wandb"
  "agent.upload_model=False"
  "agent.wandb_project=${SP_TRACKING_WANDB_PROJECT:-tracking_bfm}"
  "agent.save_interval=${SP_TRACKING_SAVE_INTERVAL:-2000}"
  "log_root=${SP_TRACKING_LOG_ROOT:-logs/rsl_rl}"
  "++task.command.command.adaptive_pre_failure_sample_window_steps=100"
  "++task.command.command.adaptive_bin_snapshot_interval_iterations=1"
  "++task.command.command.adaptive_bin_snapshot_num_buckets=2048"
  "++task.command.command.motion_scan_backend=fd"
  "++task.command.command.motion_scan_fd_executable=fdfind"
  "++task.command.command.motion_scan_workers=32"
  "++task.command.command.motion_metadata_read_backend=process"
  "++task.command.command.motion_metadata_read_workers=32"
  "++task.command.command.motion_metadata_read_chunksize=128"
)

if [[ "${HAS_TASK_OVERRIDE}" == false ]]; then
  cmd+=("task_id=${TASK_ID}")
fi

RUN_NAME="${SP_TRACKING_RUN_NAME:-trydebug_multigpu}"
if [[ -n "${RUN_NAME}" ]]; then
  cmd+=("agent.run_name=${RUN_NAME}")
fi

cd "${REPO_ROOT}"
exec "${cmd[@]}" "$@"
