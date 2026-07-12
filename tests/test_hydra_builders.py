from pathlib import Path

import yaml
from hydra import compose, initialize_config_module
from mjlab.asset_zoo.robots import G1_ACTION_SCALE

from sp_tracking.config.build_agent import build_agent_cfg, serialize_agent_cfg
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
  assert env_cfg.commands["motion"].history_steps == 0
  assert env_cfg.commands["motion"].future_steps == 1
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
  assert env_cfg.commands["motion"].history_steps == 0
  assert env_cfg.commands["motion"].future_steps == 1
  assert list(env_cfg.observations) == ["policy", "priv", "priv_critic"]
  assert list(env_cfg.observations["policy"].terms.keys())[0] == "boot_indicator_state_obs"
  assert env_cfg.commands["motion"].rewind.enabled is True
  assert env_cfg.commands["motion"].rewind.failure_probability == 0.4
  assert env_cfg.commands["motion"].adaptive_sampling.random_probability == 0.4
  assert env_cfg.commands["motion"].adaptive_sampling.strategy == "branch"
  assert "root_pos_tracking" in env_cfg.rewards
  assert "body_z_termination" in env_cfg.terminations
  assert env_cfg.commands["motion"].motion_type == "mujoco"
  assert env_cfg.commands["motion"].fk_from_joint_pos is True
  assert env_cfg.commands["motion"].termination_warmup_steps == 5
  assert type(env_cfg.actions["joint_pos"]).__name__ == "MotionTrackingJointPositionActionCfg"
  assert [sensor.name for sensor in env_cfg.scene.sensors] == ["contact_forces"]
  assert env_cfg.sim.nconmax == 200
  assert env_cfg.sim.njmax == 2048
  assert env_cfg.sim.contact_sensor_maxmatch == 128
  assert env_cfg.commands["motion"].reset_root_lift_height == 0.04
  assert env_cfg.commands["motion"].reset_min_body_z == 0.0
  assert env_cfg.commands["motion"].reset_joint_vel_limit == 10.0
  zero_six_dof_range = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
  }
  assert env_cfg.commands["motion"].pose_range == zero_six_dof_range
  assert env_cfg.commands["motion"].velocity_range == zero_six_dof_range
  assert env_cfg.commands["motion"].joint_position_range == (0.0, 0.0)
  assert env_cfg.actions["joint_pos"].scale[".*_hip_pitch_joint"] == 0.5
  assert env_cfg.actions["joint_pos"].scale[".*_wrist_pitch_joint"] == 1.0
  assert env_cfg.actions["joint_pos"].torque_limit_scale_range == (4.0, 1.0)
  assert env_cfg.actions["joint_pos"].raw_action_clip == 10.0
  assert env_cfg.actions["joint_pos"].boot_delay_steps == 2
  assert env_cfg.actions["joint_pos"].curriculum_mode == "full"
  assert env_cfg.events == {}
  assert "motion_tracking_progress" in env_cfg.curriculum
  assert env_cfg.metrics["substep_tracking_cache"].per_substep is True


def test_sp_variant_does_not_inherit_tracking_bfm_task_yaml() -> None:
  task_yaml = Path("src/sp_tracking/conf/task/tracking_bfm_sp.yaml")
  raw = yaml.safe_load(task_yaml.read_text())

  assert "tracking_bfm" not in raw["defaults"]
  assert {"command": "largedataset"} in raw["defaults"]
  assert {"obs": "sp_tracking"} in raw["defaults"]
  assert {"reward": "sp_tracking"} in raw["defaults"]


def test_sp_variant_supports_multimotion_command_override() -> None:
  cfg = _compose("task=tracking_bfm_sp", "task/command=multimotion")

  env_cfg = build_env_cfg(cfg.task)

  assert isinstance(env_cfg.commands["motion"], MultiMotionCommandCfg)
  assert env_cfg.commands["motion"].history_steps == 0
  assert env_cfg.commands["motion"].future_steps == 1
  assert env_cfg.commands["motion"].motion_type == "mujoco"
  assert env_cfg.commands["motion"].fk_from_joint_pos is True
  assert env_cfg.commands["motion"].sampling_mode == "adaptive"


def test_sp_rewind_is_yaml_switchable() -> None:
  cfg = _compose("task=tracking_bfm_sp", "task.command.command.rewind.enabled=false")

  env_cfg = build_env_cfg(cfg.task)

  assert env_cfg.commands["motion"].rewind.enabled is False


def test_sp_observation_module_adds_its_contact_sensor_in_a_mixed_ablation() -> None:
  cfg = _compose("task=tracking_bfm", "task/obs=sp_tracking")

  env_cfg = build_env_cfg(cfg.task)

  # The old reward still needs self-collision, while SP observations need feet
  # contacts.  Sensor selection follows active modules rather than task name.
  assert [sensor.name for sensor in env_cfg.scene.sensors] == [
    "contact_forces",
    "self_collision",
  ]


def test_observation_nan_policy_is_hydra_configurable() -> None:
  cfg = _compose(
    "task=tracking_bfm_sp",
    "++task.obs.observations.policy.nan_policy=error",
    "++task.obs.observations.policy.nan_check_per_term=true",
  )

  env_cfg = build_env_cfg(cfg.task)

  assert env_cfg.observations["policy"].nan_policy == "error"
  assert env_cfg.observations["policy"].nan_check_per_term is True


def test_tracking_bfm_keeps_original_events_and_action() -> None:
  cfg = _compose("task=tracking_bfm")

  env_cfg = build_env_cfg(cfg.task)

  assert type(env_cfg.actions["joint_pos"]).__name__ == "JointPositionActionCfg"
  assert [sensor.name for sensor in env_cfg.scene.sensors] == ["self_collision"]
  assert env_cfg.sim.nconmax == 128
  assert env_cfg.sim.njmax == 512
  assert env_cfg.sim.contact_sensor_maxmatch == 64
  assert env_cfg.commands["motion"].fk_from_joint_pos is False
  assert env_cfg.commands["motion"].reset_root_lift_height == 0.0
  assert env_cfg.commands["motion"].reset_min_body_z is None
  assert env_cfg.commands["motion"].reset_joint_vel_limit is None
  assert env_cfg.actions["joint_pos"].scale == G1_ACTION_SCALE
  assert "base_mass" in env_cfg.events
  assert "motor_params_implicit" not in env_cfg.events
  assert env_cfg.curriculum == {}


def test_tracking_bfm_largedataset_matches_old_tracking_task() -> None:
  cfg = _compose(
    "task=tracking_bfm_largedataset",
    "++task.command.command.motion_manifest_file=/tmp/manifest.txt",
    "++task.command.command.adaptive_bin_snapshot_interval_iterations=1",
    "++task.command.command.adaptive_bin_snapshot_num_buckets=123",
  )

  env_cfg = build_env_cfg(cfg.task)
  motion_cmd = env_cfg.commands["motion"]

  assert isinstance(motion_cmd, LargeDatasetMotionCommandCfg)
  assert motion_cmd.history_steps == 0
  assert motion_cmd.future_steps == 1
  assert motion_cmd.motion_type == "isaaclab"
  assert motion_cmd.fk_from_joint_pos is False
  assert motion_cmd.reset_root_lift_height == 0.0
  assert motion_cmd.reset_min_body_z is None
  assert motion_cmd.reset_joint_vel_limit is None
  assert motion_cmd.motion_manifest_file == "/tmp/manifest.txt"
  assert motion_cmd.motion_scan_backend == "auto"
  assert motion_cmd.motion_scan_workers == 0
  assert motion_cmd.motion_scan_fd_executable == "fd"
  assert motion_cmd.adaptive_bin_snapshot_interval_iterations == 1
  assert motion_cmd.adaptive_bin_snapshot_num_buckets == 123
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
  assert "motion_global_root_pos" in env_cfg.rewards
  assert "anchor_pos" in env_cfg.terminations
  robot = env_cfg.scene.entities["robot"]
  assert robot.spec_fn.__module__.startswith("sp_tracking.assets.robots.g1_tracking_bfm")


def test_tracking_bfm_largedataset_scan_config_is_hydra_configurable() -> None:
  cfg = _compose(
    "task=tracking_bfm_largedataset",
    "++task.command.command.motion_scan_backend=python",
    "++task.command.command.motion_scan_workers=32",
    "++task.command.command.motion_scan_fd_executable=fdfind",
    "++task.command.command.motion_metadata_read_workers=24",
    "++task.command.command.motion_metadata_read_backend=process",
    "++task.command.command.motion_metadata_read_chunksize=128",
  )

  env_cfg = build_env_cfg(cfg.task)
  motion_cmd = env_cfg.commands["motion"]

  assert motion_cmd.motion_scan_backend == "python"
  assert motion_cmd.motion_scan_workers == 32
  assert motion_cmd.motion_scan_fd_executable == "fdfind"
  assert motion_cmd.motion_metadata_read_workers == 24
  assert motion_cmd.motion_metadata_read_backend == "process"
  assert motion_cmd.motion_metadata_read_chunksize == 128


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


def test_agent_serialization_drops_irrelevant_mlp_constructor_fields() -> None:
  cfg = _compose("task=tracking_bfm_sp")

  agent_cfg = build_agent_cfg(cfg.agent)
  serialized = serialize_agent_cfg(agent_cfg)

  assert serialized["actor"]["hidden_dims"] == (1024, 1024, 512)
  assert "cnn_cfg" not in serialized["actor"]
  assert "rnn_hidden_dim" not in serialized["critic"]
