from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.utils.gpu import select_gpus
from mjlab.utils.torch import configure_torch_backends

from sp_tracking.config.build_agent import build_agent_cfg, serialize_agent_cfg
from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.tasks.tracking.rl import SpTrackingOnPolicyRunner
from sp_tracking.tasks.tracking.rl.checkpoints import resolve_local_checkpoint_path
from sp_tracking.tasks.tracking.task_catalog import TASK_SPECS


@dataclass
class PreparedTrainCfg:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  raw: DictConfig


TASK_NAME_BY_ID = {spec.task_id: spec.name for spec in TASK_SPECS}


def normalize_task_id_argv(argv: list[str]) -> list[str]:
  """Translate public mjlab task IDs into Hydra task-group overrides."""
  normalized = list(argv)
  for index, argument in enumerate(normalized[1:], start=1):
    override_kind: str | None = None
    task_value = argument
    if argument.startswith("task="):
      override_kind = "task"
      task_value = argument.removeprefix("task=")
    elif argument.startswith("task_id="):
      override_kind = "task_id"
      task_value = argument.removeprefix("task_id=")
    elif argument in TASK_NAME_BY_ID:
      override_kind = "positional"

    if override_kind is None:
      continue
    task_name = TASK_NAME_BY_ID.get(task_value)
    if task_name is not None:
      normalized[index] = f"task={task_name}"
    elif override_kind == "task_id":
      available = ", ".join(TASK_NAME_BY_ID)
      raise ValueError(
        f"Unknown task ID '{task_value}'. Available task IDs: {available}"
      )
  return normalized


def _get_world_size() -> int:
  return int(os.environ.get("WORLD_SIZE", "1"))


def _resolve_runtime_device(gpu_ids) -> tuple[str, int, int]:
  world_size = _get_world_size()
  if world_size > 1:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    return f"cuda:{local_rank}", rank, world_size

  selected_gpus, num_gpus = select_gpus(gpu_ids)
  if int(num_gpus) > 1:
    raise NotImplementedError(
      "sp-train no longer launches multi-GPU workers itself. "
      "Use torchrun, for example: "
      "`uv run torchrun --standalone --nproc_per_node=2 -m sp_tracking.scripts.train ...`"
    )
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    device = "cuda:0"
  os.environ.setdefault("MUJOCO_GL", "egl")
  return device, 0, 1


def _apply_motion_path(env_cfg: ManagerBasedRlEnvCfg, motion_path: str | None) -> None:
  if not motion_path:
    return
  motion_cmd = env_cfg.commands["motion"]
  if hasattr(motion_cmd, "motion_path"):
    motion_cmd.motion_path = motion_path
  elif hasattr(motion_cmd, "motion_file"):
    motion_cmd.motion_file = motion_path
  else:
    raise TypeError("Configured motion command has no motion_path or motion_file field.")


def prepare_train_cfg(cfg: DictConfig) -> PreparedTrainCfg:
  env_cfg = build_env_cfg(cfg.task)
  agent_cfg = build_agent_cfg(cfg.agent, cfg.task.get("agent_overrides"))
  seed = int(cfg.get("seed", agent_cfg.seed))
  env_cfg.seed = seed
  agent_cfg.seed = seed
  _apply_motion_path(env_cfg, cfg.get("motion_path"))
  return PreparedTrainCfg(env=env_cfg, agent=agent_cfg, raw=cfg)


def _make_log_dir(cfg: DictConfig, agent_cfg: RslRlOnPolicyRunnerCfg) -> Path:
  log_root = Path(str(cfg.get("log_root", "logs/rsl_rl")))
  log_root_path = log_root / agent_cfg.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if agent_cfg.run_name:
    log_dir_name += f"_{agent_cfg.run_name}"
  return log_root_path / log_dir_name


def _copy_launch_script_to_log_dir(
  log_dir: Path, launch_script_path: str | os.PathLike[str] | None
) -> Path | None:
  if not launch_script_path:
    return None

  source = Path(launch_script_path).expanduser()
  if not source.is_file():
    raise FileNotFoundError(f"Launch script does not exist: {source}")

  launch_dir = log_dir / "launch"
  launch_dir.mkdir(parents=True, exist_ok=True)
  target = launch_dir / source.name
  shutil.copy2(source, target)
  return target


def _save_resolved_cfg(log_dir: Path, cfg: DictConfig) -> None:
  log_dir.mkdir(parents=True, exist_ok=True)
  OmegaConf.save(cfg, log_dir / "cfg.yaml", resolve=True)
  OmegaConf.save(cfg, log_dir / "config.yaml", resolve=True)


def _serialize_checkpoint_cfg(cfg: DictConfig) -> dict[str, Any]:
  """Make a portable, fully resolved source-style ``cfg`` checkpoint field."""
  serialized = OmegaConf.to_container(cfg, resolve=True)
  if not isinstance(serialized, dict):
    raise TypeError("Expected the resolved training configuration to be a mapping.")
  return serialized


def _resolve_resume_path(
  cfg: DictConfig,
  agent_cfg: RslRlOnPolicyRunnerCfg,
) -> Path | None:
  checkpoint_path = cfg.get("checkpoint_path")
  if checkpoint_path:
    path = Path(str(checkpoint_path)).expanduser()
    if not path.is_file():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path
  if not bool(agent_cfg.resume):
    return None
  log_root = Path(str(cfg.get("log_root", "logs/rsl_rl"))) / agent_cfg.experiment_name
  return resolve_local_checkpoint_path(
    log_root=log_root,
    load_run=str(agent_cfg.load_run),
    load_checkpoint=str(agent_cfg.load_checkpoint),
  )


def run_train(cfg: DictConfig) -> None:
  prepared = prepare_train_cfg(cfg)
  device, _rank, world_size = _resolve_runtime_device(cfg.get("gpu_ids", [0]))
  configure_torch_backends()

  env = ManagerBasedRlEnv(
    cfg=prepared.env,
    device=device,
    render_mode="rgb_array" if bool(cfg.get("video", False)) else None,
  )
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=prepared.agent.clip_actions)
  total_frames = cfg.get("total_frames")
  if total_frames is not None:
    global_frames_per_iteration = (
      int(env.num_envs) * int(prepared.agent.num_steps_per_env) * int(world_size)
    )
    prepared.agent.max_iterations = max(
      int(total_frames) // max(global_frames_per_iteration, 1), 1
    )
  log_dir = _make_log_dir(cfg, prepared.agent)
  _save_resolved_cfg(log_dir, cfg)
  launch_script_artifact_path = _copy_launch_script_to_log_dir(
    log_dir, cfg.get("launch_script_path")
  )
  task_cfg = cfg.get("task", {})
  runner = SpTrackingOnPolicyRunner(
    wrapped_env,
    _asdict_dataclass(prepared.agent),
    str(log_dir),
    device,
    registry_name=cfg.get("registry_name"),
    launch_script_artifact_path=(
      str(launch_script_artifact_path) if launch_script_artifact_path else None
    ),
    debug_nonfinite_state=bool(task_cfg.get("debug_nonfinite_state", False)),
    checkpoint_cfg=OmegaConf.create(_serialize_checkpoint_cfg(cfg)),
  )
  resume_path = _resolve_resume_path(cfg, prepared.agent)
  if resume_path is not None:
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(str(resume_path), map_location=device)
  runner.learn(
    num_learning_iterations=prepared.agent.max_iterations,
    init_at_random_ep_len=bool(
      cfg.get("task", {}).get("init_at_random_ep_len", True)
    ),
  )
  wrapped_env.close()


def _asdict_dataclass(obj: Any) -> dict[str, Any]:
  from dataclasses import asdict

  if isinstance(obj, RslRlOnPolicyRunnerCfg):
    return serialize_agent_cfg(obj)
  return asdict(obj)


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def _hydra_main(cfg: DictConfig) -> None:
  run_train(cfg)


def main() -> None:
  sys.argv[:] = normalize_task_id_argv(sys.argv)
  _hydra_main()


if __name__ == "__main__":
  main()
