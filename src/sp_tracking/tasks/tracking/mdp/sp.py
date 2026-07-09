from __future__ import annotations

from typing import Any, Literal, cast

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactSensor
from sp_tracking.tasks.tracking.mdp.multi_commands import MultiMotionCommand
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  matrix_from_quat,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_mul,
)
from mjlab.utils.lab_api.string import resolve_matching_names

if False:
  from mjlab.envs import ManagerBasedRlEnv


TEACHER_STEPS = (0, 1, 2, 4, 8, 12, 16, 20, -1, -2, -4, -8)
STUDENT_STEPS = (0, 1, 2, 3, 4, 5, 6, -1, -2, -4, -8, -12, -16)
POLICY_HISTORY_STEPS = (0, 1, 2, 3, 4, 8, 12, 16, 20)
PRIV_HISTORY_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8)

SP_REQUIRED_BODY_NAMES = (
  "pelvis",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "head_mimic",
  "left_shoulder_yaw_link",
  "left_wrist_roll_link",
  "left_hand_mimic",
  "right_shoulder_yaw_link",
  "right_wrist_roll_link",
  "right_hand_mimic",
)

SP_KEYPOINT_BODY_NAMES = (
  "head_mimic",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "left_wrist_roll_link",
  "left_hand_mimic",
  "right_shoulder_yaw_link",
  "right_wrist_roll_link",
  "right_hand_mimic",
)

SP_FEET_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
SP_FEET_TOE_BODY_NAMES = (
  "left_ankle_roll_toe_link",
  "right_ankle_roll_toe_link",
)
SP_TERMINATION_BODY_NAMES = (
  "pelvis",
  "head_mimic",
  "left_hand_mimic",
  "right_hand_mimic",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)
SP_TERMINATION_KILL_FRAMES = 5


def _command(env: "ManagerBasedRlEnv", command_name: str) -> MultiMotionCommand:
  return cast(MultiMotionCommand, env.command_manager.get_term(command_name))


def _steps(horizon: Literal["teacher", "student"]) -> tuple[int, ...]:
  return STUDENT_STEPS if horizon == "student" else TEACHER_STEPS


def _step_tensor(
  env: "ManagerBasedRlEnv", steps: tuple[int, ...], base: torch.Tensor
) -> torch.Tensor:
  return base[:, None] + torch.as_tensor(steps, device=env.device, dtype=torch.long)


def _gather(
  env: "ManagerBasedRlEnv",
  command_name: str,
  field_name: str,
  steps: tuple[int, ...],
) -> torch.Tensor:
  cmd = _command(env, command_name)
  time_steps = _step_tensor(env, steps, cmd.time_steps)
  return cmd._gather_motion_field(field_name, cmd.motion_idx, time_steps)


def _gather_current(
  env: "ManagerBasedRlEnv", command_name: str, field_name: str
) -> torch.Tensor:
  return _gather(env, command_name, field_name, (0,))[:, 0]


def _root_motion(
  env: "ManagerBasedRlEnv",
  command_name: str,
  field_name: str,
  steps: tuple[int, ...],
) -> torch.Tensor:
  cmd = _command(env, command_name)
  data = _gather(env, command_name, field_name, steps)
  return data[:, :, cmd.motion_anchor_body_index]


def _root_motion_current(
  env: "ManagerBasedRlEnv", command_name: str, field_name: str
) -> torch.Tensor:
  return _root_motion(env, command_name, field_name, (0,))[:, 0]


def _rot6d(quat: torch.Tensor) -> torch.Tensor:
  return matrix_from_quat(quat)[..., :2].reshape(*quat.shape[:-1], 6)


def _quat_delta(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  if q1.shape[:-1] != q2.shape[:-1]:
    shape = torch.broadcast_shapes(q1.shape[:-1], q2.shape[:-1])
    q1 = q1.expand(*shape, 4)
    q2 = q2.expand(*shape, 4)
  w = q1[..., 0]
  xyz = -q1[..., 1:]
  q1_inv = torch.cat((w.unsqueeze(-1), xyz), dim=-1)
  return quat_mul(q1_inv, q2)


def _exp_sigma(
  error: torch.Tensor, sigma: list[float] | tuple[float, ...]
) -> torch.Tensor:
  if not sigma:
    raise ValueError("sigma must contain at least one value.")
  rewards = [torch.exp(-error / float(s)) for s in sigma]
  return sum(rewards) / len(rewards)


def _body_indices(names: tuple[str, ...], selected: tuple[str, ...]) -> list[int]:
  return [names.index(name) for name in selected]


def _basename(name: str) -> str:
  return name.split("/")[-1]


def _termination_buffer(
  env: "ManagerBasedRlEnv", cmd: MultiMotionCommand, name: str
) -> torch.Tensor:
  buffer = getattr(cmd, name, None)
  if (
    not isinstance(buffer, torch.Tensor)
    or buffer.shape != (env.num_envs,)
    or buffer.device != torch.device(env.device)
  ):
    buffer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    setattr(cmd, name, buffer)
  return buffer


def _continuous_termination(
  env: "ManagerBasedRlEnv", trigger: torch.Tensor, buffer: torch.Tensor
) -> torch.Tensor:
  trigger = trigger.reshape(env.num_envs, -1).any(dim=1)
  buffer.add_(trigger.to(buffer.dtype))
  buffer.masked_fill_(~trigger, 0)
  buffer.clamp_(max=max(int(SP_TERMINATION_KILL_FRAMES), 1))
  return buffer >= int(SP_TERMINATION_KILL_FRAMES)


def _body_z_values(
  env: "ManagerBasedRlEnv",
  command_name: str,
  body_z_terminate_patterns: tuple[str, ...],
) -> tuple[MultiMotionCommand, torch.Tensor, torch.Tensor]:
  cmd = _command(env, command_name)
  asset = env.scene["robot"]
  body_ids, body_names = asset.find_bodies(
    body_z_terminate_patterns, preserve_order=True
  )
  motion_body_names = tuple(cmd.cfg.body_names)
  motion_ids = [motion_body_names.index(_basename(name)) for name in body_names]
  motion_ids_t = torch.as_tensor(motion_ids, dtype=torch.long, device=env.device)
  asset_ids_t = torch.as_tensor(body_ids, dtype=torch.long, device=env.device)
  target_z = _gather_current(env, command_name, "body_pos_w")[:, motion_ids_t, 2]
  current_z = asset.data.body_link_pos_w[:, asset_ids_t, 2]
  return cmd, target_z, current_z


def _robot_body_indices(asset, selected: tuple[str, ...], device: str) -> torch.Tensor:
  ids = asset.find_bodies(selected, preserve_order=True)[0]
  return torch.as_tensor(ids, device=device, dtype=torch.long)


def _joint_target_ids(asset) -> torch.Tensor:
  return torch.arange(len(asset.joint_names), device=asset.data.joint_pos.device)


def _projected_gravity(quat: torch.Tensor) -> torch.Tensor:
  gravity = quat.new_tensor((0.0, 0.0, -1.0))
  return _quat_apply_inverse(quat, gravity.expand(*quat.shape[:-1], 3))


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  if quat.shape[:-1] != vec.shape[:-1]:
    quat = quat.expand(*vec.shape[:-1], 4)
  return quat_apply_inverse(quat, vec)


def _action_tensor(env: "ManagerBasedRlEnv") -> torch.Tensor:
  action_manager = env.action_manager
  term = _joint_action_term(env)
  if term is not None and hasattr(term, "applied_action"):
    return term.applied_action
  if hasattr(action_manager, "applied_action"):
    return action_manager.applied_action
  return action_manager.action


def _joint_action_term(env: "ManagerBasedRlEnv"):
  action_manager = env.action_manager
  get_term = getattr(action_manager, "get_term", None)
  if callable(get_term):
    try:
      return get_term("joint_pos")
    except KeyError:
      return None
  return None


def _event_observation(
  env: "ManagerBasedRlEnv", term_name: str, fallback: torch.Tensor
) -> torch.Tensor:
  event_manager = getattr(env, "event_manager", None)
  if event_manager is None:
    return fallback
  try:
    term_cfg = event_manager.get_term_cfg(term_name)
  except (KeyError, ValueError):
    return fallback
  observe = getattr(term_cfg.func, "observe", None)
  if not callable(observe):
    return fallback
  try:
    return observe()
  except NotImplementedError:
    return fallback


def _target_feet_standing(
  env: "ManagerBasedRlEnv", command_name: str, steps: tuple[int, ...] = (0,)
) -> torch.Tensor:
  cmd = _command(env, command_name)
  feet_motion_ids = _body_indices(tuple(cmd.cfg.body_names), SP_FEET_BODY_NAMES)
  feet_pos = _gather(env, command_name, "body_pos_w", steps)[:, :, feet_motion_ids]
  feet_vel = _gather(env, command_name, "body_lin_vel_w", steps)[:, :, feet_motion_ids]
  root_vel = _root_motion(env, command_name, "body_lin_vel_w", steps)
  root_vxy = root_vel[..., :2].norm(dim=-1, keepdim=True).clamp_min(1.0)
  feet_vxy = feet_vel[..., :2].norm(dim=-1)
  feet_vz = feet_vel[..., 2].abs()
  feet_z = feet_pos[..., 2]
  standing = (feet_z < 0.18) & (feet_vxy < 0.2 * root_vxy) & (feet_vz < 0.15 * root_vxy)
  return standing[:, 0] if len(steps) == 1 else standing


class _HistoryObservation:
  def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
    self.cfg = cfg
    self.env = env
    self.asset = env.scene["robot"]
    self.steps = tuple(int(s) for s in cfg.params.get("history_steps", (0,)))
    self.max_len = max(self.steps) + 1
    self.buffer: torch.Tensor | None = None

  def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
    raise NotImplementedError

  def _ensure_buffer(self, sample: torch.Tensor) -> None:
    if self.buffer is not None:
      return
    self.buffer = torch.zeros(
      (self.env.num_envs, self.max_len, sample.shape[-1]),
      dtype=sample.dtype,
      device=sample.device,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self.buffer is not None:
      self.buffer[env_ids] = 0.0

  def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
    sample = self._sample(env)
    self._ensure_buffer(sample)
    assert self.buffer is not None
    self.buffer = torch.roll(self.buffer, shifts=1, dims=1)
    self.buffer[:, 0] = sample
    idx = torch.as_tensor(self.steps, device=sample.device, dtype=torch.long)
    return self.buffer[:, idx].reshape(env.num_envs, -1)


class root_angvel_b_history(_HistoryObservation):
  def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
    sensor = env.scene["robot/imu_ang_vel"]
    return sensor.data


class projected_gravity_history(_HistoryObservation):
  def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _projected_gravity(self.asset.data.root_link_quat_w)


class root_linvel_b_history(_HistoryObservation):
  def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
    return self.asset.data.root_link_lin_vel_b


class joint_pos_history(_HistoryObservation):
  def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
    return self.asset.data.joint_pos


class joint_vel_history(_HistoryObservation):
  def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
    return self.asset.data.joint_vel


def boot_indicator_state_obs(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return torch.zeros((env.num_envs, 1), device=env.device)


def command_obs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  horizon: Literal["teacher", "student"] = "student",
) -> torch.Tensor:
  steps = _steps(horizon)
  root_quat = env.scene["robot"].data.root_link_quat_w.unsqueeze(1)
  future_quat = _root_motion(env, command_name, "body_quat_w", steps)
  future_pos = _root_motion(env, command_name, "body_pos_w", steps)
  pos_diff = _quat_apply_inverse(
    future_quat[:, :1], future_pos[:, 1:] - future_pos[:, :1]
  )
  quat_diff = _quat_delta(root_quat.expand_as(future_quat), future_quat)
  return torch.cat(
    (pos_diff.reshape(env.num_envs, -1), _rot6d(quat_diff).reshape(env.num_envs, -1)),
    dim=-1,
  )


def target_joint_pos_obs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  horizon: Literal["teacher", "student"],
) -> torch.Tensor:
  target = _gather(env, command_name, "joint_pos", _steps(horizon))
  current = env.scene["robot"].data.joint_pos.unsqueeze(1)
  diff = target - current
  return torch.cat(
    (target.reshape(env.num_envs, -1), diff.reshape(env.num_envs, -1)), dim=-1
  )


def target_root_z_obs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  horizon: Literal["teacher", "student"],
) -> torch.Tensor:
  return _root_motion(env, command_name, "body_pos_w", _steps(horizon))[..., 2]


def target_projected_gravity_b_obs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  horizon: Literal["teacher", "student"],
) -> torch.Tensor:
  quat = _root_motion(env, command_name, "body_quat_w", _steps(horizon))
  return _projected_gravity(quat).reshape(env.num_envs, -1)


def prev_actions(env: "ManagerBasedRlEnv", steps: int) -> torch.Tensor:
  term = _joint_action_term(env)
  get_recent = getattr(term, "get_recent_action_obs", None)
  if callable(get_recent):
    return get_recent(steps).reshape(env.num_envs, -1)
  action = env.action_manager.action
  prev = env.action_manager.prev_action
  if steps <= 1:
    return action
  repeated = [action, prev]
  repeated.extend([torch.zeros_like(action) for _ in range(max(steps - 2, 0))])
  return torch.cat(repeated[:steps], dim=-1)


def target_pos_b_obs(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  root_pos = env.scene["robot"].data.root_link_pos_w.unsqueeze(1)
  root_quat = env.scene["robot"].data.root_link_quat_w.unsqueeze(1)
  target_pos = _root_motion(env, command_name, "body_pos_w", TEACHER_STEPS)
  return _quat_apply_inverse(root_quat, target_pos - root_pos).reshape(env.num_envs, -1)


def target_rot_b_obs(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  root_quat = env.scene["robot"].data.root_link_quat_w.unsqueeze(1)
  target_quat = _root_motion(env, command_name, "body_quat_w", TEACHER_STEPS)
  return _rot6d(_quat_delta(root_quat.expand_as(target_quat), target_quat)).reshape(
    env.num_envs, -1
  )


def target_linvel_b_obs(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  root_quat = env.scene["robot"].data.root_link_quat_w.unsqueeze(1)
  target = _root_motion(env, command_name, "body_lin_vel_w", TEACHER_STEPS)
  return _quat_apply_inverse(root_quat, target).reshape(env.num_envs, -1)


def target_angvel_b_obs(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  root_quat = env.scene["robot"].data.root_link_quat_w.unsqueeze(1)
  target = _root_motion(env, command_name, "body_ang_vel_w", TEACHER_STEPS)
  return _quat_apply_inverse(root_quat, target).reshape(env.num_envs, -1)


class _KeypointObservation:
  def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
    self.cfg = cfg
    self.env = env
    self.asset = env.scene["robot"]
    self.asset_ids = _robot_body_indices(self.asset, SP_KEYPOINT_BODY_NAMES, env.device)

  def _motion_ids(self, command_name: str) -> list[int]:
    cmd = _command(self.env, command_name)
    return _body_indices(tuple(cmd.cfg.body_names), SP_KEYPOINT_BODY_NAMES)

  def _current(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root_pos = self.asset.data.root_link_pos_w
    root_quat = self.asset.data.root_link_quat_w
    root_lin = self.asset.data.root_link_lin_vel_w
    root_ang = self.asset.data.root_link_ang_vel_w
    pos = self.asset.data.body_link_pos_w[:, self.asset_ids]
    quat = self.asset.data.body_link_quat_w[:, self.asset_ids]
    lin = self.asset.data.body_link_lin_vel_w[:, self.asset_ids]
    ang = self.asset.data.body_link_ang_vel_w[:, self.asset_ids]
    pos_b = _quat_apply_inverse(root_quat.unsqueeze(1), pos - root_pos.unsqueeze(1))
    quat_b = _quat_delta(root_quat.unsqueeze(1).expand_as(quat), quat)
    lin_b = _quat_apply_inverse(root_quat.unsqueeze(1), lin - root_lin.unsqueeze(1))
    ang_b = _quat_apply_inverse(root_quat.unsqueeze(1), ang - root_ang.unsqueeze(1))
    return pos_b, quat_b, lin_b, ang_b


class current_keypoint_pos_b_obs(_KeypointObservation):
  def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
    return self._current()[0].reshape(env.num_envs, -1)


class current_keypoint_rot_b_obs(_KeypointObservation):
  def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
    return _rot6d(self._current()[1]).reshape(env.num_envs, -1)


class current_keypoint_linvel_b_obs(_KeypointObservation):
  def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
    return self._current()[2].reshape(env.num_envs, -1)


class current_keypoint_angvel_b_obs(_KeypointObservation):
  def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
    return self._current()[3].reshape(env.num_envs, -1)


class target_keypoints_pos_b_obs(_KeypointObservation):
  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    required_steps: int,
    include_diff: bool,
  ) -> torch.Tensor:
    steps = TEACHER_STEPS[:required_steps]
    ids = self._motion_ids(command_name)
    target_w = _gather(env, command_name, "body_pos_w", steps)[:, :, ids]
    root_pos = _root_motion(env, command_name, "body_pos_w", steps)
    root_quat = _root_motion(env, command_name, "body_quat_w", steps)
    target_b = _quat_apply_inverse(
      root_quat.unsqueeze(2), target_w - root_pos.unsqueeze(2)
    )
    if not include_diff:
      return target_b.reshape(env.num_envs, -1)
    diff = target_b - self._current()[0].unsqueeze(1)
    return torch.cat(
      (target_b.reshape(env.num_envs, -1), diff.reshape(env.num_envs, -1)), dim=-1
    )


class target_keypoints_rot_b_obs(_KeypointObservation):
  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    required_steps: int,
    include_diff: bool,
  ) -> torch.Tensor:
    steps = TEACHER_STEPS[:required_steps]
    ids = self._motion_ids(command_name)
    target_w = _gather(env, command_name, "body_quat_w", steps)[:, :, ids]
    root_quat = _root_motion(env, command_name, "body_quat_w", steps)
    target_b = _quat_delta(root_quat.unsqueeze(2).expand_as(target_w), target_w)
    target_rot = _rot6d(target_b)
    if not include_diff:
      return target_rot.reshape(env.num_envs, -1)
    diff = _rot6d(_quat_delta(self._current()[1].unsqueeze(1), target_b))
    return torch.cat(
      (target_rot.reshape(env.num_envs, -1), diff.reshape(env.num_envs, -1)), dim=-1
    )


def applied_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return _action_tensor(env)


def applied_torque(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return env.scene["robot"].data.actuator_force


def body_z_termination_obs(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  cmd, target_z, current_z = _body_z_values(
    env, command_name, SP_TERMINATION_BODY_NAMES
  )
  buffer = _termination_buffer(env, cmd, "_body_z_termination_buffer").float().unsqueeze(1)
  return torch.cat((current_z, target_z, buffer), dim=-1)


def gravity_dir_termination_obs(
  env: "ManagerBasedRlEnv", command_name: str
) -> torch.Tensor:
  cmd = _command(env, command_name)
  robot_g = _projected_gravity(env.scene["robot"].data.root_link_quat_w)
  motion_g = _projected_gravity(_root_motion_current(env, command_name, "body_quat_w"))
  buffer = _termination_buffer(
    env, cmd, "_gravity_dir_termination_buffer"
  ).float().unsqueeze(1)
  return torch.cat((robot_g, motion_g, buffer), dim=-1)


def body_z_termination(
  env: "ManagerBasedRlEnv",
  command_name: str,
  body_z_terminate_thres: tuple[float, float],
  body_z_terminate_patterns: tuple[str, ...],
) -> torch.Tensor:
  low, high = body_z_terminate_thres
  cmd, target_z, current_z = _body_z_values(
    env, command_name, body_z_terminate_patterns
  )
  target_z_min_thres = target_z + float(low)
  target_z_max_thres = target_z + float(high)
  target_z_min = target_z.amin(dim=1, keepdim=True)
  lower_relax = ((target_z_min - 0.1) / 0.2).clamp(0.0, 1.0) * 0.2
  target_z_min_thres = target_z_min_thres - lower_relax
  exceed = (current_z < target_z_min_thres) | (current_z > target_z_max_thres)
  buffer = _termination_buffer(env, cmd, "_body_z_termination_buffer")
  return _continuous_termination(env, exceed, buffer)


def gravity_dir_termination(
  env: "ManagerBasedRlEnv",
  command_name: str,
  gravity_terminate_thres: float,
) -> torch.Tensor:
  cmd = _command(env, command_name)
  obs = gravity_dir_termination_obs(env, command_name)
  robot_g = obs[:, :3]
  motion_g = obs[:, 3:6]
  exceed = torch.norm(robot_g - motion_g, dim=-1) > float(gravity_terminate_thres)
  buffer = _termination_buffer(env, cmd, "_gravity_dir_termination_buffer")
  return _continuous_termination(env, exceed, buffer)


def feet_contact_state(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if sensor.data.force_history is not None:
    force = sensor.data.force_history.mean(dim=2)
  assert force is not None
  contact_time = sensor.data.current_contact_time
  air_time = sensor.data.current_air_time
  assert contact_time is not None and air_time is not None
  in_contact = (contact_time > env.physics_dt).float()
  denom = 33.341142 * 9.81
  return torch.cat(
    (
      (force / denom).clamp(-10.0, 10.0).reshape(env.num_envs, -1),
      in_contact,
      contact_time,
      air_time,
    ),
    dim=-1,
  )


def target_feet_contact_state_obs(
  env: "ManagerBasedRlEnv", command_name: str
) -> torch.Tensor:
  return _target_feet_standing(env, command_name).float()


def domain_motor_params_implicit(env: "ManagerBasedRlEnv") -> torch.Tensor:
  joints = len(env.scene["robot"].joint_names)
  fallback = torch.ones((env.num_envs, joints * 3), device=env.device)
  return _event_observation(env, "motor_params_implicit", fallback)


def domain_perturb_body_materials(env: "ManagerBasedRlEnv") -> torch.Tensor:
  fallback = torch.ones((env.num_envs, 3), device=env.device)
  return _event_observation(env, "perturb_body_materials", fallback)


def domain_random_joint_offset(env: "ManagerBasedRlEnv") -> torch.Tensor:
  fallback = torch.zeros(
    (env.num_envs, len(env.scene["robot"].joint_names)), device=env.device
  )
  return _event_observation(env, "random_joint_offset", fallback)


def domain_perturb_gravity(env: "ManagerBasedRlEnv") -> torch.Tensor:
  fallback = torch.tensor((0.0, 0.0, -9.81), device=env.device).repeat(env.num_envs, 1)
  return _event_observation(env, "perturb_gravity", fallback)


class _RewardBase:
  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.cfg = cfg
    self.env = env
    self.asset = env.scene["robot"]


class _KeypointReward(_RewardBase):
  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.asset_ids = _robot_body_indices(self.asset, SP_KEYPOINT_BODY_NAMES, env.device)

  def _motion_ids(self, command_name: str) -> list[int]:
    cmd = _command(self.env, command_name)
    return _body_indices(tuple(cmd.cfg.body_names), SP_KEYPOINT_BODY_NAMES)

  def _current(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root_pos = self.asset.data.root_link_pos_w
    root_quat = self.asset.data.root_link_quat_w
    root_lin = self.asset.data.root_link_lin_vel_w
    root_ang = self.asset.data.root_link_ang_vel_w
    pos = self.asset.data.body_link_pos_w[:, self.asset_ids]
    quat = self.asset.data.body_link_quat_w[:, self.asset_ids]
    lin = self.asset.data.body_link_lin_vel_w[:, self.asset_ids]
    ang = self.asset.data.body_link_ang_vel_w[:, self.asset_ids]
    pos_b = _quat_apply_inverse(root_quat.unsqueeze(1), pos - root_pos.unsqueeze(1))
    quat_b = _quat_delta(root_quat.unsqueeze(1).expand_as(quat), quat)
    lin_b = _quat_apply_inverse(root_quat.unsqueeze(1), lin - root_lin.unsqueeze(1))
    ang_b = _quat_apply_inverse(root_quat.unsqueeze(1), ang - root_ang.unsqueeze(1))
    return pos_b, quat_b, lin_b, ang_b


class root_pos_tracking(_RewardBase):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    target = _root_motion_current(env, command_name, "body_pos_w")
    error = (target - self.asset.data.root_link_pos_w).norm(dim=-1, keepdim=True)
    return _exp_sigma(error, sigma).squeeze(-1)


class root_rot_tracking(_RewardBase):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    target = _root_motion_current(env, command_name, "body_quat_w")
    error = quat_error_magnitude(self.asset.data.root_link_quat_w, target).unsqueeze(-1)
    return _exp_sigma(error, sigma).squeeze(-1)


class root_vel_tracking(_RewardBase):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    current = _quat_apply_inverse(
      self.asset.data.root_link_quat_w, self.asset.data.root_link_lin_vel_w
    )
    target_quat = _root_motion_current(env, command_name, "body_quat_w")
    target = _quat_apply_inverse(
      target_quat, _root_motion_current(env, command_name, "body_lin_vel_w")
    )
    return _exp_sigma((target - current).norm(dim=-1, keepdim=True), sigma).squeeze(-1)


class root_ang_vel_tracking(_RewardBase):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    current = _quat_apply_inverse(
      self.asset.data.root_link_quat_w, self.asset.data.root_link_ang_vel_w
    )
    target_quat = _root_motion_current(env, command_name, "body_quat_w")
    target = _quat_apply_inverse(
      target_quat, _root_motion_current(env, command_name, "body_ang_vel_w")
    )
    return _exp_sigma((target - current).norm(dim=-1, keepdim=True), sigma).squeeze(-1)


class keypoint_pos_tracking(_KeypointReward):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    ids = self._motion_ids(command_name)
    root_pos = _root_motion_current(env, command_name, "body_pos_w")
    root_quat = _root_motion_current(env, command_name, "body_quat_w")
    target_w = _gather_current(env, command_name, "body_pos_w")[:, ids]
    target = _quat_apply_inverse(
      root_quat.unsqueeze(1), target_w - root_pos.unsqueeze(1)
    )
    error = (target - self._current()[0]).norm(dim=-1).mean(dim=-1, keepdim=True)
    return _exp_sigma(error, sigma).squeeze(-1)


class keypoint_vel_tracking(_KeypointReward):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    ids = self._motion_ids(command_name)
    root_quat = _root_motion_current(env, command_name, "body_quat_w")
    root_vel = _root_motion_current(env, command_name, "body_lin_vel_w")
    target_w = _gather_current(env, command_name, "body_lin_vel_w")[:, ids]
    target = _quat_apply_inverse(
      root_quat.unsqueeze(1), target_w - root_vel.unsqueeze(1)
    )
    error = (target - self._current()[2]).norm(dim=-1).mean(dim=-1, keepdim=True)
    return _exp_sigma(error, sigma).squeeze(-1)


class keypoint_rot_tracking(_KeypointReward):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    ids = self._motion_ids(command_name)
    root_quat = _root_motion_current(env, command_name, "body_quat_w")
    target_w = _gather_current(env, command_name, "body_quat_w")[:, ids]
    target = _quat_delta(root_quat.unsqueeze(1).expand_as(target_w), target_w)
    error = axis_angle_from_quat(_quat_delta(self._current()[1], target)).norm(dim=-1)
    error = error.mean(dim=-1, keepdim=True)
    return _exp_sigma(error, sigma).squeeze(-1)


class keypoint_angvel_tracking(_KeypointReward):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    ids = self._motion_ids(command_name)
    root_quat = _root_motion_current(env, command_name, "body_quat_w")
    root_ang = _root_motion_current(env, command_name, "body_ang_vel_w")
    target_w = _gather_current(env, command_name, "body_ang_vel_w")[:, ids]
    target = _quat_apply_inverse(
      root_quat.unsqueeze(1), target_w - root_ang.unsqueeze(1)
    )
    error = (target - self._current()[3]).norm(dim=-1).mean(dim=-1, keepdim=True)
    return _exp_sigma(error, sigma).squeeze(-1)


class joint_pos_tracking(_RewardBase):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    target = _gather_current(env, command_name, "joint_pos")
    error = (target - self.asset.data.joint_pos).abs().mean(dim=-1, keepdim=True)
    return _exp_sigma(error, sigma).squeeze(-1)


class joint_vel_tracking(_RewardBase):
  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
  ) -> torch.Tensor:
    target = _gather_current(env, command_name, "joint_vel")
    error = (target - self.asset.data.joint_vel).abs().mean(dim=-1, keepdim=True)
    return _exp_sigma(error, sigma).squeeze(-1)


def survival(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def joint_vel_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return -env.scene["robot"].data.joint_vel.square().sum(dim=-1)


def action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
  term = _joint_action_term(env)
  get_recent = getattr(term, "get_recent_action_rate_actions", None)
  if callable(get_recent):
    action_buf = get_recent(2)
    diff = action_buf[:, 0] - action_buf[:, 1]
    return -diff.square().sum(dim=-1)
  diff = env.action_manager.action - env.action_manager.prev_action
  return -diff.square().sum(dim=-1)


class feet_air_time_ref(_RewardBase):
  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.reward_time = torch.zeros(
      env.num_envs, len(SP_FEET_BODY_NAMES), device=env.device
    )
    self.last_contact = torch.zeros_like(self.reward_time, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.reward_time[env_ids] = 0.0
    self.last_contact[env_ids] = False

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    thres: float,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time
    assert contact_time is not None
    current = contact_time > env.physics_dt
    first_contact = (~self.last_contact) & current
    self.last_contact[:] = current
    target = _target_feet_standing(env, command_name)
    mismatch = target ^ current
    self.reward_time += torch.where(mismatch, -env.step_dt, env.step_dt)
    reward = ((self.reward_time - float(thres)).clamp_max(0.0) * first_contact).sum(1)
    self.reward_time *= ~current
    return reward


class feet_air_time_ref_dense(_RewardBase):
  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.feet_ids = _robot_body_indices(self.asset, SP_FEET_BODY_NAMES, env.device)
    self.toe_ids = _robot_body_indices(self.asset, SP_FEET_TOE_BODY_NAMES, env.device)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    air_h_low: float,
    air_h_high: float,
    contact_h_low: float,
    contact_h_high: float,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time
    assert contact_time is not None
    current = contact_time > env.physics_dt
    target = _target_feet_standing(env, command_name)
    mismatch = current ^ target
    both_air = (~current) & (~target)
    both_contact = current & target
    penalty = torch.zeros_like(current, dtype=torch.float32)
    penalty[mismatch] = -1.0

    feet_z = self.asset.data.body_link_pos_w[:, self.feet_ids, 2]
    toe_z = self.asset.data.body_link_pos_w[:, self.toe_ids, 2]
    air_height = torch.minimum(feet_z, toe_z)
    air_span = max(float(air_h_high) - float(air_h_low), 1.0e-6)
    air_ratio = ((air_height - float(air_h_low)) / air_span).clamp(0.0, 1.0)
    penalty = torch.where(both_air, -(1.0 - air_ratio), penalty)

    contact_height = torch.maximum(feet_z, toe_z)
    contact_span = max(float(contact_h_high) - float(contact_h_low), 1.0e-6)
    contact_ratio = ((contact_height - float(contact_h_low)) / contact_span).clamp(
      0.0, 1.0
    )
    penalty = torch.where(both_contact, -contact_ratio, penalty)
    return penalty.mean(dim=1)


class joint_pos_limits(_RewardBase):
  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    joint_names = cfg.params.get("joint_names", ".*")
    ids, _ = resolve_matching_names(joint_names, self.asset.joint_names)
    self.joint_ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)

  def __call__(
    self, env: "ManagerBasedRlEnv", soft_factor: float, **_: Any
  ) -> torch.Tensor:
    limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
    mean = (limits[..., 0] + limits[..., 1]) * 0.5
    span = limits[..., 1] - limits[..., 0]
    lower = mean - 0.5 * span * float(soft_factor)
    upper = mean + 0.5 * span * float(soft_factor)
    pos = self.asset.data.joint_pos[:, self.joint_ids]
    violation = (lower - pos).clamp_min(0.0) + (pos - upper).clamp_min(0.0)
    return -violation.sum(dim=1) / max(1.0 - float(soft_factor), 1.0e-6)


class joint_torque_limits(_RewardBase):
  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    joint_names = cfg.params.get("joint_names", ".*")
    _, names = resolve_matching_names(joint_names, self.asset.joint_names)
    actuator_names = list(self.asset.actuator_names)
    self.act_ids = torch.as_tensor(
      [actuator_names.index(name) for name in names],
      device=env.device,
      dtype=torch.long,
    )

  def __call__(
    self, env: "ManagerBasedRlEnv", soft_factor: float, **_: Any
  ) -> torch.Tensor:
    force_range = env.sim.model.actuator_forcerange[:, self.asset.indexing.ctrl_ids]
    limits = torch.maximum(force_range[..., 0].abs(), force_range[..., 1].abs())
    soft_limits = limits[:, self.act_ids].clamp_min(1.0e-6) * float(soft_factor)
    torque = self.asset.data.actuator_force[:, self.act_ids]
    high = (torque / soft_limits - 1.0).clamp_min(0.0)
    low = (-torque / soft_limits - 1.0).clamp_min(0.0)
    return -(high + low).sum(dim=1)
