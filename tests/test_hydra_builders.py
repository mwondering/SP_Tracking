from hydra import compose, initialize_config_module

from sp_tracking.config.build_agent import build_agent_cfg
from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  MotionCommandCfg as LargeDatasetMotionCommandCfg,
)
from sp_tracking.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)


def _compose(*overrides: str):
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=list(overrides))


def test_default_tracking_bfm_builds_multimotion_cfg() -> None:
  cfg = _compose()

  env_cfg = build_env_cfg(cfg.task)

  assert isinstance(env_cfg.commands["motion"], MultiMotionCommandCfg)
  assert list(env_cfg.observations.keys()) == ["actor", "critic"]
  assert list(env_cfg.observations["actor"].terms.keys()) == [
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "body_pos",
    "body_ori",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  ]
  assert env_cfg.observations["actor"].terms["joint_pos"].params == {}
  assert "motion_global_root_pos" in env_cfg.rewards
  assert "anchor_pos" in env_cfg.terminations
  assert "base_mass" in env_cfg.events


def test_sp_variant_builds_largedataset_cfg() -> None:
  cfg = _compose("task=tracking_bfm_sp")

  env_cfg = build_env_cfg(cfg.task)

  assert isinstance(env_cfg.commands["motion"], LargeDatasetMotionCommandCfg)
  assert list(env_cfg.observations["actor"].terms.keys())[0] == "boot_indicator_state_obs"
  assert "root_pos_tracking" in env_cfg.rewards
  assert "body_z_termination" in env_cfg.terminations
  assert env_cfg.commands["motion"].motion_type == "mujoco"


def test_tracking_bfm_command_adaptive_window_is_hydra_configurable() -> None:
  cfg = _compose("++task.command.command.adaptive_pre_failure_sample_window_steps=123")

  env_cfg = build_env_cfg(cfg.task)

  assert env_cfg.commands["motion"].adaptive_pre_failure_sample_window_steps == 123


def test_agent_cfg_matches_tracking_bfm_defaults() -> None:
  cfg = _compose()

  agent_cfg = build_agent_cfg(cfg.agent)

  assert agent_cfg.experiment_name == "g1_tracking"
  assert agent_cfg.num_steps_per_env == 24
  assert agent_cfg.actor.hidden_dims == (2048, 2048, 1024, 1024, 512, 256, 128)
  assert agent_cfg.algorithm.learning_rate == 1.0e-3
