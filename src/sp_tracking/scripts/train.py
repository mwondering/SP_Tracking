from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.utils.gpu import select_gpus

from sp_tracking.config.build_agent import build_agent_cfg
from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.tasks.tracking.rl import MotionTrackingOnPolicyRunner


@dataclass
class PreparedTrainCfg:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  raw: DictConfig


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
  agent_cfg = build_agent_cfg(cfg.agent)
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


def run_train(cfg: DictConfig) -> None:
  prepared = prepare_train_cfg(cfg)
  selected_gpus, num_gpus = select_gpus(cfg.get("gpu_ids", [0]))
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    device = "cuda:0"
  os.environ.setdefault("MUJOCO_GL", "egl")

  if int(num_gpus) > 1:
    raise NotImplementedError(
      "sp-train currently supports direct single-process launch. "
      "Use torchrunx integration after the single-GPU path is validated."
    )

  env = ManagerBasedRlEnv(
    cfg=prepared.env,
    device=device,
    render_mode="rgb_array" if bool(cfg.get("video", False)) else None,
  )
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=prepared.agent.clip_actions)
  log_dir = _make_log_dir(cfg, prepared.agent)
  runner = MotionTrackingOnPolicyRunner(
    wrapped_env,
    _asdict_dataclass(prepared.agent),
    str(log_dir),
    device,
    registry_name=cfg.get("registry_name"),
  )
  runner.learn(
    num_learning_iterations=prepared.agent.max_iterations,
    init_at_random_ep_len=True,
  )
  wrapped_env.close()


def _asdict_dataclass(obj: Any) -> dict[str, Any]:
  from dataclasses import asdict

  return asdict(obj)


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
  run_train(cfg)


if __name__ == "__main__":
  main()
