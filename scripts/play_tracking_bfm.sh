#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/play_tracking_bfm.sh --checkpoint-file PATH [--motion-file PATH|--motion-path PATH] [options]

Options:
  --task TASK_NAME  # legacy checkpoint only; see README task table
  --checkpoint-file PATH
  --motion-file PATH
  --motion-path PATH
  --num-envs N
  --viewer native|viser
  --domain-randomization true|false
  --dry-run
  -h, --help
USAGE
}

TASK="${SP_TRACKING_PLAY_TASK:-}"
CHECKPOINT_FILE="${SP_TRACKING_CHECKPOINT_FILE:-}"
MOTION_FILE="${SP_TRACKING_MOTION_FILE:-}"
MOTION_PATH="${SP_TRACKING_MOTION_PATH:-}"
NUM_ENVS="${SP_TRACKING_PLAY_NUM_ENVS:-1}"
VIEWER="${SP_TRACKING_PLAY_VIEWER:-viser}"
DOMAIN_RANDOMIZATION="${SP_TRACKING_DOMAIN_RANDOMIZATION:-}"
DRY_RUN="${SP_TRACKING_DRY_RUN:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --checkpoint-file) CHECKPOINT_FILE="$2"; shift 2 ;;
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

if [[ -n "${TASK}" ]]; then
  case "${TASK}" in
    tracking_bfm|\
    tracking_bfm_sp|\
    tracking_bfm_sp_ablation_bfm_actor|\
    tracking_bfm_sp_ablation_student_actor|\
    tracking_bfm_sp_ablation_teacher_actor|\
    tracking_bfm_student_actor_bfm_critic|\
    tracking_bfm_teacher_actor_bfm_critic|\
    tracking_bfm_wbteleop_actor_bfm_critic|\
    tracking_bfm_wbteleop_actor_heft_critic|\
    tracking_bfm_heft_reward|\
    tracking_bfm_sp_ablation_bfm_actor_heft_reward|\
    tracking_bfm_sp_ablation_student_actor_heft_reward|\
    tracking_bfm_sp_ablation_teacher_actor_heft_reward|\
    tracking_bfm_student_actor_bfm_critic_heft_reward|\
    tracking_bfm_teacher_actor_bfm_critic_heft_reward|\
    tracking_bfm_wbteleop_actor_bfm_critic_heft_reward|\
    tracking_bfm_wbteleop_actor_heft_critic_heft_reward|\
    tracking_bfm_spv1_actor_heft_critic_heft_reward|\
    tracking_bfm_spv2_actor_heft_critic_heft_reward|\
    tracking_bfm_spv3_actor_heft_critic_heft_reward|\
    tracking_bfm_spv4_actor_heft_critic_heft_reward) ;;
    *) echo "Invalid task: ${TASK}" >&2; exit 2 ;;
  esac
fi

case "${VIEWER}" in
  native|viser) ;;
  *) echo "Invalid viewer: ${VIEWER}" >&2; exit 2 ;;
esac

cmd=(
  uv run sp-play
  --checkpoint-file "${CHECKPOINT_FILE}"
  --num-envs "${NUM_ENVS}"
  --viewer "${VIEWER}"
)

if [[ -z "${CHECKPOINT_FILE}" ]]; then
  echo "Provide --checkpoint-file." >&2
  exit 2
fi
[[ -f "${CHECKPOINT_FILE}" ]] || { echo "Checkpoint file not found: ${CHECKPOINT_FILE}" >&2; exit 2; }

if [[ -n "${TASK}" ]]; then
  cmd+=(--task "${TASK}")
fi

case "${DOMAIN_RANDOMIZATION}" in
  "") ;;
  true) cmd+=(--domain-randomization True) ;;
  false) cmd+=(--domain-randomization False) ;;
  *) echo "Invalid domain randomization value: ${DOMAIN_RANDOMIZATION}" >&2; exit 2 ;;
esac

if [[ -n "${MOTION_FILE}" && -n "${MOTION_PATH}" ]]; then
  echo "Provide either --motion-file or --motion-path, not both." >&2
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
