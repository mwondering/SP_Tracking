from __future__ import annotations

from hydra import compose, initialize_config_module

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg


def _compose(task_name: str):
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=[f"task={task_name}"])


def test_named_variants_form_a_supported_profile_matrix() -> None:
  expectations = {
    "tracking_bfm": {
      "environment": "bfm",
      "reference": "bfm",
      "obs_groups": ("actor", "critic"),
      "reward": "motion_global_root_pos",
      "agent_groups": {"actor": ("actor",), "critic": ("critic",)},
      "sp_environment": False,
    },
    "tracking_bfm_sp": {
      "environment": "sp_tracking",
      "reference": "sp_tracking",
      "obs_groups": ("policy", "priv", "priv_critic"),
      "reward": "root_pos_tracking",
      "agent_groups": {
        "actor": ("policy",),
        "critic": ("policy", "priv", "priv_critic"),
      },
      "sp_environment": True,
    },
    "tracking_bfm_sp_old_obs_old_reward_bfm_agent": {
      "environment": "sp_tracking",
      "reference": "sp_tracking",
      "obs_groups": ("actor", "critic"),
      "reward": "motion_global_root_pos",
      "agent_groups": {"actor": ("actor",), "critic": ("critic",)},
      "sp_environment": True,
    },
    "tracking_bfm_sp_old_reward": {
      "environment": "sp_tracking",
      "reference": "sp_tracking",
      "obs_groups": ("policy", "priv", "priv_critic"),
      "reward": "motion_global_root_pos",
      "agent_groups": {
        "actor": ("policy",),
        "critic": ("policy", "priv", "priv_critic"),
      },
      "sp_environment": True,
    },
    "tracking_bfm_sp_bfm_agent_old_reward": {
      "environment": "sp_tracking",
      "reference": "sp_tracking",
      "obs_groups": ("policy", "priv", "priv_critic"),
      "reward": "motion_global_root_pos",
      "agent_groups": {
        "actor": ("policy",),
        "critic": ("policy", "priv", "priv_critic"),
      },
      "sp_environment": True,
    },
    "tracking_bfm_sp_bfm_agent_old_obs": {
      "environment": "sp_tracking",
      "reference": "sp_tracking",
      "obs_groups": ("actor", "critic"),
      "reward": "root_pos_tracking",
      "agent_groups": {"actor": ("actor",), "critic": ("critic",)},
      "sp_environment": True,
    },
  }

  for task_name, expected in expectations.items():
    cfg = _compose(task_name)
    prepared = prepare_train_cfg(cfg)
    env_cfg = build_env_cfg(cfg.task)

    assert cfg.task.variant.environment_profile == expected["environment"]
    assert cfg.task.variant.reference_profile == expected["reference"]
    assert tuple(env_cfg.observations) == expected["obs_groups"]
    assert expected["reward"] in env_cfg.rewards
    assert prepared.agent.obs_groups == expected["agent_groups"]

    if expected["sp_environment"]:
      robot = env_cfg.scene.entities["robot"]
      assert robot.spec_fn.__module__.startswith(
        "sp_tracking.assets.robots.g1_sp_tracking"
      )
      command = env_cfg.commands["motion"]
      assert command.motion_type == "mujoco"
      assert command.fk_from_joint_pos is True
      assert command.recompute_joint_vel_from_joint_pos is True
      assert type(env_cfg.actions["joint_pos"]).__name__ == "SpTrackingJointPositionActionCfg"
      assert type(command).__name__ == "LargeDatasetMultiMotionCommandCfg"
      assert command.adaptive_sampling.strategy == "branch"
      assert command.rewind.enabled is True
      assert "sp_tracking_progress" in env_cfg.curriculum
      assert "substep_tracking_cache" in env_cfg.metrics
      assert {"body_z_termination", "gravity_dir_termination"} <= set(
        env_cfg.terminations
      )
