from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  MotionCommandCfg as LargeDatasetMotionCommandCfg,
)
from sp_tracking.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from sp_tracking.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def test_tracking_entrypoint_registers_default_tasks() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  tasks = list_tasks()

  assert registry.TRACKING_BFM_TASK_ID in tasks
  assert registry.TRACKING_BFM_SP_TASK_ID in tasks


def test_registered_default_task_loads_hydra_built_configs() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  env_cfg = load_env_cfg(registry.TRACKING_BFM_TASK_ID)
  rl_cfg = load_rl_cfg(registry.TRACKING_BFM_TASK_ID)

  assert isinstance(env_cfg.commands["motion"], MultiMotionCommandCfg)
  assert rl_cfg.experiment_name == "g1_tracking"
  assert load_runner_cls(registry.TRACKING_BFM_TASK_ID) is MotionTrackingOnPolicyRunner


def test_registered_sp_task_uses_large_dataset_command() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  env_cfg = load_env_cfg(registry.TRACKING_BFM_SP_TASK_ID)

  assert isinstance(env_cfg.commands["motion"], LargeDatasetMotionCommandCfg)
