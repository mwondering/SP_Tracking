from __future__ import annotations

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from sp_tracking.config.build_agent import build_agent_cfg
from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.tasks.tracking.rl import SpTrackingOnPolicyRunner


TRACKING_BFM_TASK_ID = "SPTracking-G1-BFM"
TRACKING_BFM_LARGEDATASET_TASK_ID = "SPTracking-G1-BFM-LargeDataset"
TRACKING_BFM_SP_TASK_ID = "SPTracking-G1-BFM-SP"


def _compose_train(overrides: list[str] | None = None) -> DictConfig:
  overrides = overrides or []
  if GlobalHydra.instance().is_initialized():
    return compose(config_name="train", overrides=overrides)
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=overrides)


def _register_task(task_id: str, overrides: list[str] | None = None) -> None:
  if task_id in list_tasks():
    return
  cfg = _compose_train(overrides)
  play_cfg = _compose_train([*(overrides or []), "task.num_envs=1"])
  register_mjlab_task(
    task_id=task_id,
    env_cfg=build_env_cfg(cfg.task),
    play_env_cfg=build_env_cfg(play_cfg.task),
    rl_cfg=build_agent_cfg(cfg.agent, cfg.task.get("agent_overrides")),
    runner_cls=SpTrackingOnPolicyRunner,
  )


_register_task(TRACKING_BFM_TASK_ID)
_register_task(
  TRACKING_BFM_LARGEDATASET_TASK_ID,
  ["task=tracking_bfm_largedataset"],
)
_register_task(TRACKING_BFM_SP_TASK_ID, ["task=tracking_bfm_sp"])


__all__ = [
  "TRACKING_BFM_TASK_ID",
  "TRACKING_BFM_LARGEDATASET_TASK_ID",
  "TRACKING_BFM_SP_TASK_ID",
]
