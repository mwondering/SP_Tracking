from __future__ import annotations

from collections import OrderedDict
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from sp_tracking.assets.robots import (
  get_g1_motion_tracking_robot_cfg,
  get_g1_tracking_bfm_robot_cfg,
)
from sp_tracking.tasks.tracking import mdp
from sp_tracking.tasks.tracking.mdp.actions import MotionTrackingJointPositionActionCfg
from sp_tracking.tasks.tracking.mdp import randomizations as sp_randomizations
from sp_tracking.tasks.tracking.mdp import sp as sp_mdp
from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  MotionCommandCfg as LargeDatasetMotionCommandCfg,
)
from sp_tracking.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)


OBS_TERMS = {
  "generated_commands": mdp.generated_commands,
  "motion_anchor_pos_b": mdp.motion_anchor_pos_b,
  "motion_anchor_ori_b": mdp.motion_anchor_ori_b,
  "robot_body_pos_b": mdp.robot_body_pos_b,
  "robot_body_ori_b": mdp.robot_body_ori_b,
  "builtin_sensor": mjlab_mdp.builtin_sensor,
  "joint_pos_rel": mjlab_mdp.joint_pos_rel,
  "joint_vel_rel": mjlab_mdp.joint_vel_rel,
  "last_action": mjlab_mdp.last_action,
  "boot_indicator_state_obs": sp_mdp.boot_indicator_state_obs,
  "command_obs": sp_mdp.command_obs,
  "target_joint_pos_obs": sp_mdp.target_joint_pos_obs,
  "target_root_z_obs": sp_mdp.target_root_z_obs,
  "target_projected_gravity_b_obs": sp_mdp.target_projected_gravity_b_obs,
  "target_pos_b_obs": sp_mdp.target_pos_b_obs,
  "target_rot_b_obs": sp_mdp.target_rot_b_obs,
  "target_linvel_b_obs": sp_mdp.target_linvel_b_obs,
  "target_angvel_b_obs": sp_mdp.target_angvel_b_obs,
  "current_keypoint_pos_b_obs": sp_mdp.current_keypoint_pos_b_obs,
  "current_keypoint_rot_b_obs": sp_mdp.current_keypoint_rot_b_obs,
  "current_keypoint_linvel_b_obs": sp_mdp.current_keypoint_linvel_b_obs,
  "current_keypoint_angvel_b_obs": sp_mdp.current_keypoint_angvel_b_obs,
  "target_keypoints_pos_b_obs": sp_mdp.target_keypoints_pos_b_obs,
  "target_keypoints_rot_b_obs": sp_mdp.target_keypoints_rot_b_obs,
  "root_linvel_b_history": sp_mdp.root_linvel_b_history,
  "root_angvel_b_history": sp_mdp.root_angvel_b_history,
  "projected_gravity_history": sp_mdp.projected_gravity_history,
  "joint_pos_history": sp_mdp.joint_pos_history,
  "joint_vel_history": sp_mdp.joint_vel_history,
  "prev_actions": sp_mdp.prev_actions,
  "applied_action": sp_mdp.applied_action,
  "applied_torque": sp_mdp.applied_torque,
  "body_z_termination_obs": sp_mdp.body_z_termination_obs,
  "gravity_dir_termination_obs": sp_mdp.gravity_dir_termination_obs,
  "feet_contact_state": sp_mdp.feet_contact_state,
  "target_feet_contact_state_obs": sp_mdp.target_feet_contact_state_obs,
  "domain_motor_params_implicit": sp_mdp.domain_motor_params_implicit,
  "domain_perturb_body_materials": sp_mdp.domain_perturb_body_materials,
  "domain_random_joint_offset": sp_mdp.domain_random_joint_offset,
  "domain_perturb_gravity": sp_mdp.domain_perturb_gravity,
}

REWARD_TERMS = {
  "motion_global_anchor_position_error_exp": mdp.motion_global_anchor_position_error_exp,
  "motion_global_anchor_orientation_error_exp": mdp.motion_global_anchor_orientation_error_exp,
  "motion_relative_body_position_error_exp": mdp.motion_relative_body_position_error_exp,
  "motion_relative_body_orientation_error_exp": mdp.motion_relative_body_orientation_error_exp,
  "motion_global_body_linear_velocity_error_exp": mdp.motion_global_body_linear_velocity_error_exp,
  "motion_global_body_angular_velocity_error_exp": mdp.motion_global_body_angular_velocity_error_exp,
  "action_rate_l2": mdp.action_rate_l2,
  "joint_action_rate_l2": mdp.joint_action_rate_l2,
  "joint_pos_limits": mdp.joint_pos_limits,
  "self_collision_cost": mdp.self_collision_cost,
  "root_pos_tracking": sp_mdp.root_pos_tracking,
  "root_rot_tracking": sp_mdp.root_rot_tracking,
  "root_vel_tracking": sp_mdp.root_vel_tracking,
  "root_ang_vel_tracking": sp_mdp.root_ang_vel_tracking,
  "keypoint_pos_tracking": sp_mdp.keypoint_pos_tracking,
  "keypoint_vel_tracking": sp_mdp.keypoint_vel_tracking,
  "keypoint_rot_tracking": sp_mdp.keypoint_rot_tracking,
  "keypoint_angvel_tracking": sp_mdp.keypoint_angvel_tracking,
  "joint_pos_tracking": sp_mdp.joint_pos_tracking,
  "joint_vel_tracking": sp_mdp.joint_vel_tracking,
  "survival": sp_mdp.survival,
  "joint_vel_l2": sp_mdp.joint_vel_l2,
  "action_rate_l2_sp": sp_mdp.action_rate_l2,
  "feet_air_time_ref": sp_mdp.feet_air_time_ref,
  "feet_air_time_ref_dense": sp_mdp.feet_air_time_ref_dense,
  "joint_pos_limits_sp": sp_mdp.joint_pos_limits,
  "joint_torque_limits": sp_mdp.joint_torque_limits,
}

TERMINATION_TERMS = {
  "time_out": mjlab_mdp.time_out,
  "bad_anchor_pos_z_only": mdp.bad_anchor_pos_z_only,
  "bad_anchor_ori": mdp.bad_anchor_ori,
  "bad_motion_body_pos_z_only": mdp.bad_motion_body_pos_z_only,
  "body_z_termination": sp_mdp.body_z_termination,
  "gravity_dir_termination": sp_mdp.gravity_dir_termination,
}

EVENT_TERMS = {
  "perturb_body_com": sp_randomizations.perturb_body_com,
  "perturb_body_materials": sp_randomizations.perturb_body_materials,
  "motor_params_implicit": sp_randomizations.motor_params_implicit,
  "random_joint_offset": sp_randomizations.random_joint_offset,
  "perturb_root_vel": sp_randomizations.perturb_root_vel,
  "perturb_body_wrench": sp_randomizations.perturb_body_wrench,
  "perturb_gravity": sp_randomizations.perturb_gravity,
}

CURRICULUM_TERMS = {
  "motion_tracking_progress": sp_randomizations.motion_tracking_progress,
}


def _to_container(value: Any) -> Any:
  if isinstance(value, (DictConfig, ListConfig)):
    return OmegaConf.to_container(value, resolve=True)
  return value


def _to_tuple(value: Any) -> tuple[Any, ...]:
  value = _to_container(value)
  if value is None:
    return ()
  if isinstance(value, tuple):
    return value
  if isinstance(value, list):
    return tuple(value)
  return (value,)


def _optional_int(value: Any) -> int | None:
  value = _to_container(value)
  if value is None:
    return None
  return int(value)


def _optional_auto_float(value: Any) -> float | str | None:
  value = _to_container(value)
  if value is None or value == "auto":
    return value
  return float(value)


def _params(raw: Any | None) -> dict[str, Any]:
  params = _to_container(raw) if raw is not None else {}
  params = dict(params)
  for key, value in list(params.items()):
    if key == "asset_cfg" and isinstance(value, dict):
      params[key] = _scene_entity_cfg(value)
    elif isinstance(value, list):
      params[key] = tuple(value)
    elif isinstance(value, dict):
      params[key] = {
        nested_key: tuple(nested_value) if isinstance(nested_value, list) else nested_value
        for nested_key, nested_value in value.items()
      }
  return params


def _scene_entity_cfg(data: dict[str, Any]) -> SceneEntityCfg:
  kwargs = dict(data)
  entity_name = kwargs.pop("entity_name")
  for key, value in list(kwargs.items()):
    if isinstance(value, list):
      kwargs[key] = tuple(value)
  return SceneEntityCfg(entity_name, **kwargs)


def _noise(raw: Any | None) -> UniformNoiseCfg | None:
  if raw is None:
    return None
  n_min, n_max = _to_container(raw)
  return UniformNoiseCfg(n_min=float(n_min), n_max=float(n_max))


def _build_observations(cfg: DictConfig) -> dict[str, ObservationGroupCfg]:
  obs_cfg = cfg.observations if "observations" in cfg else cfg.obs.observations
  groups = OrderedDict()
  for group_name, group_cfg in obs_cfg.items():
    terms = OrderedDict()
    for item in group_cfg.terms:
      term_name = str(item.name)
      term_key = str(item.term)
      terms[term_name] = ObservationTermCfg(
        func=OBS_TERMS[term_key],
        params=_params(item.get("params")),
        noise=_noise(item.get("noise")),
        history_length=int(item.get("history_length", 0)),
      )
    groups[group_name] = ObservationGroupCfg(
      terms=terms,
      concatenate_terms=bool(group_cfg.get("concatenate_terms", True)),
      enable_corruption=bool(group_cfg.get("enable_corruption", False)),
    )
  return groups


def _build_rewards(cfg: DictConfig) -> dict[str, RewardTermCfg]:
  reward_cfg = cfg.rewards if "rewards" in cfg else cfg.reward.rewards
  rewards = OrderedDict()
  for item in reward_cfg:
    rewards[str(item.name)] = RewardTermCfg(
      func=REWARD_TERMS[str(item.term)],
      weight=float(item.weight),
      params=_params(item.get("params")),
    )
  return rewards


def _build_terminations(cfg: DictConfig) -> dict[str, TerminationTermCfg]:
  terminations = OrderedDict()
  for item in cfg.terminations:
    terminations[str(item.name)] = TerminationTermCfg(
      func=TERMINATION_TERMS[str(item.term)],
      params=_params(item.get("params")),
      time_out=bool(item.get("time_out", False)),
    )
  return terminations


def _build_command(cfg: DictConfig):
  command_cfg = cfg.command.command if "command" in cfg.command else cfg.command
  kwargs = {
    "entity_name": str(command_cfg.entity_name),
    "motion_path": str(command_cfg.get("motion_path", "")),
    "motion_file": str(command_cfg.get("motion_file", "")),
    "extra_reference_motion_file": str(command_cfg.get("extra_reference_motion_file", "")),
    "motion_type": str(command_cfg.motion_type),
    "fk_from_joint_pos": bool(command_cfg.get("fk_from_joint_pos", False)),
    "anchor_body_name": str(command_cfg.anchor_body_name),
    "body_names": _to_tuple(command_cfg.body_names),
    "pose_range": _params(command_cfg.get("pose_range")),
    "velocity_range": _params(command_cfg.get("velocity_range")),
    "joint_position_range": tuple(command_cfg.joint_position_range),
    "future_steps": int(command_cfg.get("future_steps", 5)),
    "history_steps": int(command_cfg.get("history_steps", 5)),
    "adaptive_uniform_ratio": float(command_cfg.get("adaptive_uniform_ratio", 0.1)),
    "adaptive_bin_width_s": float(command_cfg.get("adaptive_bin_width_s", 1.0)),
    "adaptive_bin_width_steps": _optional_int(
      command_cfg.get("adaptive_bin_width_steps")
    ),
    "adaptive_init_num_failures": float(
      command_cfg.get("adaptive_init_num_failures", 1.0)
    ),
    "adaptive_failure_rate_window_iterations": _optional_int(
      command_cfg.get("adaptive_failure_rate_window_iterations")
    ),
    "adaptive_failure_rate_window_chunks": int(
      command_cfg.get("adaptive_failure_rate_window_chunks", 40)
    ),
    "adaptive_failure_rate_max_over_mean": float(
      command_cfg.get("adaptive_failure_rate_max_over_mean", 200.0)
    ),
    "adaptive_sequence_length_agnostic": bool(
      command_cfg.get("adaptive_sequence_length_agnostic", True)
    ),
    "adaptive_max_prob_per_bin": _optional_auto_float(
      command_cfg.get("adaptive_max_prob_per_bin", "auto")
    ),
    "adaptive_max_prob_per_motion": _optional_auto_float(
      command_cfg.get("adaptive_max_prob_per_motion", "auto")
    ),
    "adaptive_pre_failure_sample_window_steps": int(
      command_cfg.get("adaptive_pre_failure_sample_window_steps", 200)
    ),
    "sampling_mode": str(command_cfg.get("sampling_mode", "adaptive")),
    "if_log_metrics": bool(command_cfg.get("if_log_metrics", True)),
    "resampling_time_range": tuple(command_cfg.resampling_time_range),
    "debug_vis": bool(command_cfg.get("debug_vis", True)),
  }
  if command_cfg.type == "large_dataset":
    kwargs.update(
      {
        "active_subset_size": int(command_cfg.get("active_subset_size", 20_000)),
        "subset_refresh_count": int(command_cfg.get("subset_refresh_count", 10)),
        "subset_min_resident_iterations": int(
          command_cfg.get("subset_min_resident_iterations", 50)
        ),
        "subset_adaptive_refresh_ratio": float(
          command_cfg.get("subset_adaptive_refresh_ratio", 0.5)
        ),
        "subset_adaptive_candidate_pool_size": int(
          command_cfg.get("subset_adaptive_candidate_pool_size", 10_000)
        ),
        "adaptive_bin_pool_reset_interval_iterations": int(
          command_cfg.get("adaptive_bin_pool_reset_interval_iterations", 5000)
        ),
        "adaptive_bin_snapshot_interval_iterations": int(
          command_cfg.get("adaptive_bin_snapshot_interval_iterations", 0)
        ),
        "adaptive_bin_snapshot_num_buckets": int(
          command_cfg.get("adaptive_bin_snapshot_num_buckets", 2048)
        ),
        "adaptive_bin_snapshot_dir": str(
          command_cfg.get("adaptive_bin_snapshot_dir", "")
        ),
        "motion_manifest_file": str(command_cfg.get("motion_manifest_file", "")),
        "motion_metadata_cache_file": str(
          command_cfg.get("motion_metadata_cache_file", "")
        ),
        "motion_metadata_cache_wait_timeout_s": float(
          command_cfg.get("motion_metadata_cache_wait_timeout_s", 7200.0)
        ),
        "motion_metadata_cache_poll_interval_s": float(
          command_cfg.get("motion_metadata_cache_poll_interval_s", 0.25)
        ),
        "motion_manifest_wait_timeout_s": float(
          command_cfg.get("motion_manifest_wait_timeout_s", 600.0)
        ),
        "motion_manifest_poll_interval_s": float(
          command_cfg.get("motion_manifest_poll_interval_s", 0.25)
        ),
        "motion_scan_log_interval_s": float(
          command_cfg.get("motion_scan_log_interval_s", 10.0)
        ),
      }
    )
    return LargeDatasetMotionCommandCfg(**kwargs)
  if command_cfg.type == "multi":
    return MultiMotionCommandCfg(**kwargs)
  raise ValueError(f"Unsupported command type: {command_cfg.type}")


def _build_events(cfg: DictConfig) -> dict[str, EventTermCfg]:
  command_cfg = cfg.command.command if "command" in cfg.command else cfg.command
  velocity_range = _params(command_cfg.velocity_range)
  robot = cfg.robot
  events = OrderedDict()
  if cfg.events.get("push_robot") and cfg.events.push_robot.enabled:
    events["push_robot"] = EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=tuple(cfg.events.push_robot.interval_range_s),
      params={"velocity_range": velocity_range},
    )
  if cfg.events.get("base_com") and cfg.events.base_com.enabled:
    events["base_com"] = EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=_to_tuple(robot.base_com_body_names)),
        "operation": "add",
        "ranges": _params(cfg.events.base_com.ranges),
      },
    )
  if cfg.events.get("base_mass") and cfg.events.base_mass.enabled:
    events["base_mass"] = EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=_to_tuple(cfg.events.base_mass.body_names)
        ),
        "operation": "add",
        "ranges": tuple(cfg.events.base_mass.ranges),
      },
    )
  if cfg.events.get("encoder_bias") and cfg.events.encoder_bias.enabled:
    events["encoder_bias"] = EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": tuple(cfg.events.encoder_bias.bias_range),
      },
    )
  if cfg.events.get("foot_friction") and cfg.events.foot_friction.enabled:
    events["foot_friction"] = EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=str(robot.foot_geom_pattern)),
        "operation": "abs",
        "ranges": tuple(cfg.events.foot_friction.ranges),
        "shared_random": bool(cfg.events.foot_friction.shared_random),
      },
    )
  for name, event_cfg in cfg.events.items():
    if name in {"push_robot", "base_com", "base_mass", "encoder_bias", "foot_friction"}:
      continue
    if not event_cfg.get("enabled", True):
      continue
    events[str(name)] = EventTermCfg(
      mode=str(event_cfg.mode),
      func=EVENT_TERMS[str(event_cfg.term)],
      interval_range_s=(
        tuple(event_cfg.interval_range_s) if event_cfg.get("interval_range_s") else None
      ),
      is_global_time=bool(event_cfg.get("is_global_time", False)),
      min_step_count_between_reset=int(
        event_cfg.get("min_step_count_between_reset", 0)
      ),
      params=_params(event_cfg.get("params")),
    )
  return events


def _build_action(cfg: DictConfig):
  scale = G1_ACTION_SCALE if cfg.action.scale == "g1_action_scale" else cfg.action.scale
  action_type = str(cfg.action.get("type", "joint_position"))
  cfg_cls = (
    MotionTrackingJointPositionActionCfg
    if action_type == "motion_tracking"
    else JointPositionActionCfg
  )
  extra_kwargs = {}
  if cfg_cls is MotionTrackingJointPositionActionCfg:
    extra_kwargs = {
      "max_delay": int(cfg.action.get("max_delay", 2)),
      "delay_full_progress": float(cfg.action.get("delay_full_progress", 0.8)),
      "alpha": tuple(cfg.action.get("alpha", (0.8, 1.0))),
      "torque_limit_scale_range": tuple(
        cfg.action.get("torque_limit_scale_range", (1.0, 1.0))
      ),
      "torque_limit_progress_range": tuple(
        cfg.action.get("torque_limit_progress_range", (0.0, 0.8))
      ),
    }
  return {
    "joint_pos": cfg_cls(
      entity_name=str(cfg.robot.entity_name),
      actuator_names=_to_tuple(cfg.action.actuator_names),
      scale=scale,
      use_default_offset=bool(cfg.action.use_default_offset),
      **extra_kwargs,
    )
  }


def _build_curriculum(cfg: DictConfig) -> dict[str, CurriculumTermCfg]:
  terms = OrderedDict()
  for name, term_cfg in cfg.get("curriculum", {}).items():
    if not term_cfg.get("enabled", True):
      continue
    terms[str(name)] = CurriculumTermCfg(
      func=CURRICULUM_TERMS[str(term_cfg.term)],
      params=_params(term_cfg.get("params")),
    )
  return terms


def _build_sensors(cfg: DictConfig):
  sensors = [
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(
        mode="subtree",
        pattern=str(cfg.robot.self_collision_primary_pattern),
        entity="robot",
      ),
      secondary=ContactMatch(
        mode="subtree",
        pattern=str(cfg.robot.self_collision_primary_pattern),
        entity="robot",
      ),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    )
  ]
  if cfg.name.endswith("_sp"):
    sensors.append(
      ContactSensorCfg(
        name="contact_forces",
        primary=ContactMatch(
          mode="subtree",
          pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        global_frame=True,
        num_slots=1,
        track_air_time=True,
        history_length=3,
      )
    )
  return tuple(sensors)


def _build_robot(cfg: DictConfig):
  asset = str(cfg.robot.get("asset", "tracking_bfm_g1"))
  if asset == "tracking_bfm_g1":
    return get_g1_tracking_bfm_robot_cfg()
  if asset == "motion_tracking_g1":
    return get_g1_motion_tracking_robot_cfg()
  raise ValueError(f"Unsupported robot asset: {asset}")


def build_env_cfg(cfg: DictConfig | dict[str, Any]) -> ManagerBasedRlEnvCfg:
  if not isinstance(cfg, DictConfig):
    cfg = OmegaConf.create(cfg)
  cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
  scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type=str(cfg.scene.terrain_type)),
    num_envs=int(cfg.num_envs),
    env_spacing=float(cfg.scene.env_spacing),
    entities={"robot": _build_robot(cfg)},
    sensors=_build_sensors(cfg),
  )
  viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name=str(cfg.robot.viewer_body_name),
    distance=float(cfg.viewer.distance),
    fovy=float(cfg.viewer.fovy),
    elevation=float(cfg.viewer.elevation),
    azimuth=float(cfg.viewer.azimuth),
  )
  sim = SimulationCfg(
    nconmax=int(cfg.sim.nconmax),
    njmax=int(cfg.sim.njmax),
    mujoco=MujocoCfg(
      timestep=float(cfg.sim.timestep),
      iterations=int(cfg.sim.iterations),
      ls_iterations=int(cfg.sim.ls_iterations),
    ),
  )
  return ManagerBasedRlEnvCfg(
    decimation=int(cfg.decimation),
    scene=scene,
    observations=_build_observations(cfg),
    actions=_build_action(cfg),
    commands={"motion": _build_command(cfg)},
    events=_build_events(cfg),
    rewards=_build_rewards(cfg),
    terminations=_build_terminations(cfg),
    curriculum=_build_curriculum(cfg),
    viewer=viewer,
    sim=sim,
    episode_length_s=float(cfg.episode_length_s),
    seed=int(cfg.seed),
  )
