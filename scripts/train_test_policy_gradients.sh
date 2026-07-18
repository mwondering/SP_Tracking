#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LAUNCH_SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
TASK_ID="SPTracking-G1-TestPolicyGradients"

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/train_test_policy_gradients.sh SIMPLE_NPZ HARD_NPZ [hydra_overrides...]

  scripts/train_test_policy_gradients.sh \
    [uv run sp-train] \
    task_id=SPTracking-G1-TestPolicyGradients \
    task.gradient_test.mode=mixed \
    task.gradient_test.simple_motion_file=/absolute/path/simple.npz \
    task.gradient_test.hard_motion_file=/absolute/path/hard.npz \
    agent.run_name=gradient_mixed \
    [hydra_overrides...]

The supplied mode is accepted for command compatibility but the launcher owns
it: three concurrent runs are always started as simple, hard, and mixed.
An agent.run_name ending in _simple, _hard, or _mixed is treated as a prefix;
the three final run names receive those suffixes automatically.

Environment:
  SP_TRACKING_GRADIENT_GPU_GROUPS="0;1;2"
      Three semicolon-separated GPU groups, one per experiment. A group can
      contain multiple comma-separated GPUs, for example "0,1;2,3;4,5".
  SP_TRACKING_GRADIENT_RUN_PREFIX=gradient
      Run-name prefix used when agent.run_name is not supplied.
  SP_TRACKING_SIMPLE_MOTION_FILE=/path/simple.npz
  SP_TRACKING_HARD_MOTION_FILE=/path/hard.npz
      Alternatives to positional paths or Hydra path overrides.
  SP_TRACKING_ALLOW_SHARED_GPUS=0
      Set to 1 only when overlapping GPU groups are intentional.
  SP_TRACKING_DRY_RUN=0
      Set to 1 to validate inputs and print all three commands without running.

Examples:
  scripts/train_test_policy_gradients.sh simple.npz hard.npz agent.seed=42

  SP_TRACKING_GRADIENT_GPU_GROUPS="0;1;2" \
    scripts/train_test_policy_gradients.sh \
      task.gradient_test.simple_motion_file=/data/simple.npz \
      task.gradient_test.hard_motion_file=/data/hard.npz \
      agent.run_name=pilot_mixed \
      task.num_envs=16384

  SP_TRACKING_GRADIENT_GPU_GROUPS="0,1;2,3;4,5" \
    scripts/train_test_policy_gradients.sh simple.npz hard.npz
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Also accept the tokens copied from an existing `uv run sp-train ...`
# command, so the remaining Hydra overrides can be pasted unchanged.
if [[ "${1:-}" == "uv" && "${2:-}" == "run" && "${3:-}" == "sp-train" ]]; then
  shift 3
fi

SIMPLE_MOTION_FILE="${SP_TRACKING_SIMPLE_MOTION_FILE:-}"
HARD_MOTION_FILE="${SP_TRACKING_HARD_MOTION_FILE:-}"
RUN_PREFIX="${SP_TRACKING_GRADIENT_RUN_PREFIX:-gradient}"
EXTRA_OVERRIDES=()

for argument in "$@"; do
  case "${argument}" in
    task.gradient_test.simple_motion_file=*)
      SIMPLE_MOTION_FILE="${argument#*=}"
      ;;
    task.gradient_test.hard_motion_file=*)
      HARD_MOTION_FILE="${argument#*=}"
      ;;
    task.gradient_test.mode=*)
      # The launcher deliberately replaces this once per child process.
      ;;
    agent.run_name=*)
      RUN_PREFIX="${argument#*=}"
      RUN_PREFIX="${RUN_PREFIX%_simple}"
      RUN_PREFIX="${RUN_PREFIX%_hard}"
      RUN_PREFIX="${RUN_PREFIX%_mixed}"
      ;;
    task_id=*)
      if [[ "${argument#*=}" != "${TASK_ID}" ]]; then
        echo "Expected task_id=${TASK_ID}, got ${argument}" >&2
        exit 2
      fi
      ;;
    task=*)
      if [[ "${argument#*=}" != "test_policy_gradients" ]]; then
        echo "Expected task=test_policy_gradients, got ${argument}" >&2
        exit 2
      fi
      ;;
    agent.resume=*)
      if [[ "${argument#*=}" != "false" && "${argument#*=}" != "False" ]]; then
        echo "All three diagnostic runs must start from scratch." >&2
        exit 2
      fi
      ;;
    checkpoint_path=*)
      if [[ -n "${argument#*=}" && "${argument#*=}" != "null" ]]; then
        echo "checkpoint_path is incompatible with from-scratch runs." >&2
        exit 2
      fi
      ;;
    launch_script_path=*)
      # Always archive this three-run launcher instead.
      ;;
    gpu_ids=*)
      echo "Use SP_TRACKING_GRADIENT_GPU_GROUPS instead of ${argument}." >&2
      exit 2
      ;;
    *=*)
      EXTRA_OVERRIDES+=("${argument}")
      ;;
    *)
      if [[ -z "${SIMPLE_MOTION_FILE}" ]]; then
        SIMPLE_MOTION_FILE="${argument}"
      elif [[ -z "${HARD_MOTION_FILE}" ]]; then
        HARD_MOTION_FILE="${argument}"
      else
        echo "Unexpected positional argument: ${argument}" >&2
        usage
        exit 2
      fi
      ;;
  esac
done

validate_motion_file() {
  local label="$1"
  local path="$2"
  if [[ -z "${path}" ]]; then
    echo "Missing ${label} motion NPZ path." >&2
    usage
    exit 2
  fi
  if [[ ! -f "${path}" ]]; then
    echo "${label} motion file does not exist: ${path}" >&2
    exit 2
  fi
  if [[ "${path,,}" != *.npz ]]; then
    echo "${label} motion file must end in .npz: ${path}" >&2
    exit 2
  fi
}

validate_motion_file "simple" "${SIMPLE_MOTION_FILE}"
validate_motion_file "hard" "${HARD_MOTION_FILE}"
if [[ "$(realpath -- "${SIMPLE_MOTION_FILE}")" == "$(realpath -- "${HARD_MOTION_FILE}")" ]]; then
  echo "Simple and hard motions must be different files." >&2
  exit 2
fi
if [[ -z "${RUN_PREFIX}" ]]; then
  echo "The run-name prefix must not be empty." >&2
  exit 2
fi

GPU_GROUPS="${SP_TRACKING_GRADIENT_GPU_GROUPS:-0;1;2}"
IFS=';' read -r -a GPU_GROUP_LIST <<< "${GPU_GROUPS}"
if [[ "${#GPU_GROUP_LIST[@]}" -ne 3 ]]; then
  echo "SP_TRACKING_GRADIENT_GPU_GROUPS must contain exactly three groups." >&2
  exit 2
fi

declare -A SEEN_GPUS=()
for gpu_group in "${GPU_GROUP_LIST[@]}"; do
  if [[ ! "${gpu_group}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Invalid GPU group '${gpu_group}'; expected values such as 0 or 0,1." >&2
    exit 2
  fi
  IFS=',' read -r -a group_gpus <<< "${gpu_group}"
  for gpu in "${group_gpus[@]}"; do
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" && "${SP_TRACKING_ALLOW_SHARED_GPUS:-0}" != "1" ]]; then
      echo "GPU ${gpu} occurs in multiple groups; refusing concurrent sharing." >&2
      exit 2
    fi
    SEEN_GPUS["${gpu}"]=1
  done
done

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

PIDS=()
declare -A PID_TO_MODE=()

launch_experiment() {
  local mode="$1"
  local gpu_group="$2"
  local run_name="${RUN_PREFIX}_${mode}"
  local launcher=()
  local command=()
  local pid

  IFS=',' read -r -a group_gpus <<< "${gpu_group}"
  if [[ "${#group_gpus[@]}" -eq 1 ]]; then
    launcher=(uv run sp-train)
  else
    launcher=(
      uv run torchrun
      --standalone
      "--nproc_per_node=${#group_gpus[@]}"
      -m sp_tracking.scripts.train
    )
  fi

  command=(
    "${launcher[@]}"
    "task_id=${TASK_ID}"
    "launch_script_path=${LAUNCH_SCRIPT_PATH}"
    "gpu_ids=[0]"
    "${EXTRA_OVERRIDES[@]}"
    "task.gradient_test.mode=${mode}"
    "task.gradient_test.simple_motion_file=${SIMPLE_MOTION_FILE}"
    "task.gradient_test.hard_motion_file=${HARD_MOTION_FILE}"
    "agent.run_name=${run_name}"
    "agent.resume=false"
  )

  printf '[launcher] %s GPUs=%s command:' "${mode}" "${gpu_group}"
  printf ' %q' "${command[@]}"
  printf '\n'

  if [[ "${SP_TRACKING_DRY_RUN:-0}" == "1" ]]; then
    return
  fi

  CUDA_VISIBLE_DEVICES="${gpu_group}" "${command[@]}" &
  pid=$!
  PIDS+=("${pid}")
  PID_TO_MODE["${pid}"]="${mode}"
  echo "[launcher] ${mode} started as PID ${pid}; terminal output will interleave."
}

launch_experiment simple "${GPU_GROUP_LIST[0]}"
launch_experiment hard "${GPU_GROUP_LIST[1]}"
launch_experiment mixed "${GPU_GROUP_LIST[2]}"

if [[ "${SP_TRACKING_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

terminate_running() {
  local pid
  trap - INT TERM HUP
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

on_signal() {
  echo "[launcher] Interrupted; stopping all three training processes." >&2
  terminate_running
  exit 130
}
trap on_signal INT TERM HUP

remaining="${#PIDS[@]}"
while [[ "${remaining}" -gt 0 ]]; do
  completed_pid=""
  set +e
  wait -n -p completed_pid
  status=$?
  set -e
  remaining=$((remaining - 1))
  completed_mode="${PID_TO_MODE[${completed_pid}]:-unknown}"
  if [[ "${status}" -ne 0 ]]; then
    echo "[launcher] ${completed_mode} failed with status ${status}; stopping the other runs." >&2
    terminate_running
    exit "${status}"
  fi
  echo "[launcher] ${completed_mode} completed successfully."
done

trap - INT TERM HUP
echo "[launcher] All three policy-gradient experiments completed successfully."
