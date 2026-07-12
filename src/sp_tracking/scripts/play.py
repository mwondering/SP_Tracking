from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from omegaconf import DictConfig, OmegaConf

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from sp_tracking.config.build_agent import serialize_agent_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.rl import MotionTrackingOnPolicyRunner


TASK_OVERRIDES = {
  "tracking_bfm": [],
  "tracking_bfm_largedataset": ["task=tracking_bfm_largedataset"],
  "tracking_bfm_sp": ["task=tracking_bfm_sp"],
}


@dataclass(frozen=True)
class PlayConfig:
  # New checkpoints are self-describing; task is only needed for legacy local
  # checkpoints that do not carry the source-style ``cfg`` field.
  task: Literal["tracking_bfm", "tracking_bfm_largedataset", "tracking_bfm_sp"] | None = None
  checkpoint_file: str | None = None
  motion_path: str | None = None
  motion_file: str | None = None
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["native", "viser"] = "viser"
  domain_randomization: bool | None = None
  stochastic_policy: bool = False


@dataclass
class PreparedPlayCfg:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  checkpoint_path: Path


def _compose_train(overrides: list[str]):
  from hydra import compose, initialize_config_module
  from hydra.core.global_hydra import GlobalHydra

  if GlobalHydra.instance().is_initialized():
    return compose(config_name="train", overrides=overrides)
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=overrides)


def _apply_play_motion(env_cfg: ManagerBasedRlEnvCfg, cfg: PlayConfig) -> None:
  if cfg.motion_file is not None and cfg.motion_path is not None:
    raise ValueError("Provide either motion_file or motion_path, not both.")
  motion_cmd = env_cfg.commands["motion"]
  if cfg.motion_file is not None:
    path = Path(cfg.motion_file)
    if not path.is_file():
      raise FileNotFoundError(f"Motion file not found: {path}")
    motion_cmd.motion_path = ""
    motion_cmd.motion_file = str(path)
  if cfg.motion_path is not None:
    path = Path(cfg.motion_path)
    if not path.is_dir():
      raise FileNotFoundError(f"Motion path not found: {path}")
    motion_cmd.motion_file = ""
    motion_cmd.motion_path = str(path)


def _resolve_checkpoint_path(cfg: PlayConfig) -> Path:
  if cfg.checkpoint_file is None:
    raise ValueError(
      "checkpoint_file is required. Play intentionally uses a local checkpoint; "
      "W&B is logging-only."
    )
  checkpoint_path = Path(cfg.checkpoint_file).expanduser()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
  return checkpoint_path


def _load_saved_train_cfg(checkpoint_path: Path) -> DictConfig | None:
  """Load the source-style resolved ``cfg`` embedded in a local checkpoint."""
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  raw_cfg = checkpoint.get("cfg")
  if raw_cfg is None:
    return None
  if isinstance(raw_cfg, DictConfig):
    return OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
  if isinstance(raw_cfg, Mapping):
    return OmegaConf.create(dict(raw_cfg))
  raise TypeError(
    f"Checkpoint cfg must be a mapping, got {type(raw_cfg).__name__} in {checkpoint_path}."
  )


def _prepare_checkpoint_train_cfg(
  checkpoint_path: Path, requested_task: str | None
):
  saved_cfg = _load_saved_train_cfg(checkpoint_path)
  if saved_cfg is None:
    if requested_task is None:
      raise ValueError(
        "This legacy checkpoint has no embedded cfg. Pass --task explicitly "
        "to select tracking_bfm or tracking_bfm_sp."
      )
    fallback_cfg = _compose_train(TASK_OVERRIDES[requested_task])
    return prepare_train_cfg(fallback_cfg)

  if "task" not in saved_cfg or "agent" not in saved_cfg:
    raise ValueError(
      "The checkpoint cfg is not an SP_Tracking train configuration. "
      "Raw motion_tracking checkpoints cannot be loaded by the different RSL policy stack."
    )
  saved_task = str(saved_cfg.task.get("name", ""))
  if requested_task is not None and requested_task != saved_task:
    raise ValueError(
      f"Requested task '{requested_task}' does not match checkpoint task '{saved_task}'."
    )
  return prepare_train_cfg(saved_cfg)


def prepare_play_cfg(cfg: PlayConfig) -> PreparedPlayCfg:
  checkpoint_path = _resolve_checkpoint_path(cfg)
  prepared_train = _prepare_checkpoint_train_cfg(checkpoint_path, cfg.task)
  env_cfg = prepared_train.env
  agent_cfg = prepared_train.agent
  env_cfg.scene.num_envs = int(cfg.num_envs)
  if cfg.domain_randomization is False:
    env_cfg.events = {}
  _apply_play_motion(env_cfg, cfg)
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
  runner = MotionTrackingOnPolicyRunner(
    wrapped_env,
    serialize_agent_cfg(prepared.agent),
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
