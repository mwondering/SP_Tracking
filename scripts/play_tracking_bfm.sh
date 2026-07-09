#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/play_tracking_bfm.sh --checkpoint-file PATH [--motion-file PATH|--motion-path PATH] [options]

Options:
  --task tracking_bfm|tracking_bfm_largedataset|tracking_bfm_sp
  --checkpoint-file PATH
  --wandb-run-path ENTITY/PROJECT/RUN_ID
  --wandb-checkpoint-name NAME
  --motion-file PATH
  --motion-path PATH
  --num-envs N
  --viewer native|viser
  --domain-randomization true|false
  --dry-run
  -h, --help
USAGE
}

TASK="${SP_TRACKING_PLAY_TASK:-tracking_bfm}"
CHECKPOINT_FILE="${SP_TRACKING_CHECKPOINT_FILE:-}"
WANDB_RUN_PATH="${SP_TRACKING_WANDB_RUN_PATH:-}"
WANDB_CHECKPOINT_NAME="${SP_TRACKING_WANDB_CHECKPOINT_NAME:-}"
MOTION_FILE="${SP_TRACKING_MOTION_FILE:-}"
MOTION_PATH="${SP_TRACKING_MOTION_PATH:-}"
NUM_ENVS="${SP_TRACKING_PLAY_NUM_ENVS:-1}"
VIEWER="${SP_TRACKING_PLAY_VIEWER:-viser}"
DOMAIN_RANDOMIZATION="${SP_TRACKING_DOMAIN_RANDOMIZATION:-true}"
DRY_RUN="${SP_TRACKING_DRY_RUN:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --checkpoint-file) CHECKPOINT_FILE="$2"; shift 2 ;;
    --wandb-run-path) WANDB_RUN_PATH="$2"; shift 2 ;;
    --wandb-checkpoint-name) WANDB_CHECKPOINT_NAME="$2"; shift 2 ;;
    --motion-file) MOTION_FILE="$2"; shift 2 ;;
    --motion-path) MOTION_PATH="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --viewer) VIEWER="$2"; shift 2 ;;
    --domain-randomization) DOMAIN_RANDOMIZATION="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "${TASK}" in
  tracking_bfm|tracking_bfm_largedataset|tracking_bfm_sp) ;;
  *) echo "Invalid task: ${TASK}" >&2; exit 2 ;;
esac

case "${VIEWER}" in
  native|viser) ;;
  *) echo "Invalid viewer: ${VIEWER}" >&2; exit 2 ;;
esac

cmd=(
  uv run sp-play
  --task "${TASK}"
  --num-envs "${NUM_ENVS}"
  --viewer "${VIEWER}"
  --domain-randomization "${DOMAIN_RANDOMIZATION}"
)

if [[ -n "${CHECKPOINT_FILE}" ]]; then
  [[ -f "${CHECKPOINT_FILE}" ]] || { echo "Checkpoint file not found: ${CHECKPOINT_FILE}" >&2; exit 2; }
  cmd+=(--checkpoint-file "${CHECKPOINT_FILE}")
elif [[ -n "${WANDB_RUN_PATH}" ]]; then
  cmd+=(--wandb-run-path "${WANDB_RUN_PATH}")
  if [[ -n "${WANDB_CHECKPOINT_NAME}" ]]; then
    cmd+=(--wandb-checkpoint-name "${WANDB_CHECKPOINT_NAME}")
  fi
else
  echo "Provide --checkpoint-file or --wandb-run-path." >&2
  exit 2
fi

if [[ -n "${MOTION_FILE}" ]]; then
  [[ -f "${MOTION_FILE}" ]] || { echo "Motion file not found: ${MOTION_FILE}" >&2; exit 2; }
  cmd+=(--motion-file "${MOTION_FILE}")
fi

if [[ -n "${MOTION_PATH}" ]]; then
  [[ -d "${MOTION_PATH}" ]] || { echo "Motion path not found: ${MOTION_PATH}" >&2; exit 2; }
  cmd+=(--motion-path "${MOTION_PATH}")
fi

cd "${REPO_ROOT}"
printf 'Running command: '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

exec "${cmd[@]}"
