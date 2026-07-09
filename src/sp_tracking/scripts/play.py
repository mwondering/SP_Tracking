from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from sp_tracking.config.build_agent import build_agent_cfg
from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from sp_tracking.tasks.tracking.rl.checkpoints import (
  get_wandb_checkpoint_path,
  resolve_local_checkpoint_path,
)

if False:
  from omegaconf import DictConfig


TASK_OVERRIDES = {
  "tracking_bfm": [],
  "tracking_bfm_largedataset": ["task=tracking_bfm_largedataset"],
  "tracking_bfm_sp": ["task=tracking_bfm_sp"],
}


@dataclass(frozen=True)
class PlayConfig:
  task: Literal["tracking_bfm", "tracking_bfm_largedataset", "tracking_bfm_sp"] = "tracking_bfm"
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  motion_path: str | None = None
  motion_file: str | None = None
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["native", "viser"] = "viser"
  domain_randomization: bool = True
  stochastic_policy: bool = False
  log_root: str = "logs/rsl_rl"
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"


@dataclass
class PreparedPlayCfg:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  checkpoint_path: Path | None


def _compose_train(overrides: list[str]):
  from hydra import compose, initialize_config_module
  from hydra.core.global_hydra import GlobalHydra

  if GlobalHydra.instance().is_initialized():
    return compose(config_name="train", overrides=overrides)
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=overrides)


def _apply_play_motion(env_cfg: ManagerBasedRlEnvCfg, cfg: PlayConfig) -> None:
  motion_cmd = env_cfg.commands["motion"]
  if cfg.motion_file is not None:
    path = Path(cfg.motion_file)
    if not path.is_file():
      raise FileNotFoundError(f"Motion file not found: {path}")
    motion_cmd.motion_file = str(path)
  if cfg.motion_path is not None:
    path = Path(cfg.motion_path)
    if not path.is_dir():
      raise FileNotFoundError(f"Motion path not found: {path}")
    motion_cmd.motion_path = str(path)


def prepare_play_cfg(cfg: PlayConfig) -> PreparedPlayCfg:
  train_cfg = _compose_train(TASK_OVERRIDES[cfg.task])
  env_cfg = build_env_cfg(train_cfg.task)
  agent_cfg = build_agent_cfg(train_cfg.agent)
  env_cfg.scene.num_envs = int(cfg.num_envs)
  if not cfg.domain_randomization:
    env_cfg.events = {}
  _apply_play_motion(env_cfg, cfg)

  checkpoint_path = None
  if cfg.checkpoint_file is not None:
    checkpoint_path = Path(cfg.checkpoint_file)
    if not checkpoint_path.is_file():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
  elif cfg.wandb_run_path is not None:
    log_root = Path(cfg.log_root) / agent_cfg.experiment_name
    checkpoint_path, _ = get_wandb_checkpoint_path(
      log_root=log_root,
      run_path=cfg.wandb_run_path,
      checkpoint_name=cfg.wandb_checkpoint_name,
    )
  elif train_cfg.agent.get("resume", False):
    log_root = Path(cfg.log_root) / agent_cfg.experiment_name
    checkpoint_path = resolve_local_checkpoint_path(
      log_root=log_root,
      load_run=cfg.load_run,
      load_checkpoint=cfg.load_checkpoint,
    )
  return PreparedPlayCfg(env=env_cfg, agent=agent_cfg, checkpoint_path=checkpoint_path)


def _get_trained_policy(runner: MotionTrackingOnPolicyRunner, device: str, stochastic: bool):
  if not stochastic:
    return runner.get_inference_policy(device=device)

  actor = runner.alg.get_policy()
  actor.eval()

  class StochasticPolicy:
    def __call__(self, obs):
      with torch.no_grad():
        return actor(obs, stochastic_output=True)

  return StochasticPolicy()


def run_play(cfg: PlayConfig) -> None:
  configure_torch_backends()
  prepared = prepare_play_cfg(cfg)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env = ManagerBasedRlEnv(cfg=deepcopy(prepared.env), device=device)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=prepared.agent.clip_actions)
  if prepared.checkpoint_path is None:
    raise ValueError("checkpoint_file is required for trained play.")
  runner = MotionTrackingOnPolicyRunner(
    wrapped_env,
    asdict(prepared.agent),
    log_dir=None,
    device=device,
  )
  runner.load(
    str(prepared.checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = _get_trained_policy(runner, device=device, stochastic=cfg.stochastic_policy)
  if cfg.viewer == "native":
    NativeMujocoViewer(runner.env, policy).run()
  else:
    ViserPlayViewer(runner.env, policy).run()
  runner.env.close()


def main() -> None:
  import tyro

  cfg = tyro.cli(PlayConfig)
  run_play(cfg)


if __name__ == "__main__":
  main()
