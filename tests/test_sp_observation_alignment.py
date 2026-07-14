from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import torch


sp = importlib.import_module("sp_tracking.tasks.tracking.mdp.sp")


def test_sp_keypoint_order_matches_heft_asset_order() -> None:
  assert sp.SP_KEYPOINT_BODY_NAMES == (
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "head_mimic",
    "left_shoulder_yaw_link",
    "left_wrist_roll_link",
    "left_hand_mimic",
    "right_shoulder_yaw_link",
    "right_wrist_roll_link",
    "right_hand_mimic",
  )


def test_rot6d_matches_reference_column_major_semantics():
  # A non-symmetric rotation distinguishes the source column-contiguous
  # encoding from the previous row-interleaved flattening.
  quat = torch.tensor([[0.8253356, 0.1693705, -0.2822842, 0.4516547]])
  matrix = sp.matrix_from_quat(quat)
  expected = matrix[..., :, :2].transpose(-2, -1).reshape(1, 6)

  actual = sp._rot6d(quat)

  torch.testing.assert_close(actual, expected)
  assert not torch.equal(actual, matrix[..., :, :2].reshape(1, 6))


def test_quaternion_frame_and_delta_follow_distinct_source_semantics():
  qx = torch.tensor(
    [[math.cos(0.3), math.sin(0.3), 0.0, 0.0]], dtype=torch.float32
  )
  qy = torch.tensor(
    [[math.cos(0.4), 0.0, math.sin(0.4), 0.0]], dtype=torch.float32
  )
  qx_inv = torch.cat((qx[:, :1], -qx[:, 1:]), dim=-1)

  torch.testing.assert_close(sp._quat_in_frame(qx, qy), sp.quat_mul(qx_inv, qy))
  torch.testing.assert_close(sp._quat_delta(qx, qy), sp.quat_mul(qy, qx_inv))
  assert not torch.allclose(sp._quat_in_frame(qx, qy), sp._quat_delta(qx, qy))


class _FakeAsset:
  def __init__(self, *, data: SimpleNamespace, body_names: tuple[str, ...], joint_count: int):
    self.data = data
    self.body_names = body_names
    self.joint_names = tuple(f"joint_{idx}" for idx in range(joint_count))

  def find_bodies(self, selected, preserve_order: bool = False):
    del preserve_order
    if isinstance(selected, str):
      selected = (selected,)
    ids = [self.body_names.index(name) for name in selected]
    names = [self.body_names[idx] for idx in ids]
    return ids, names


class _FakeCommand:
  def __init__(self, fields: dict[str, torch.Tensor], body_names: tuple[str, ...]):
    self.fields = fields
    self.motion_idx = torch.zeros(1, dtype=torch.long)
    self.time_steps = torch.full((1,), 5, dtype=torch.long)
    self.motion_anchor_body_index = body_names.index("pelvis")
    self.cfg = SimpleNamespace(body_names=body_names)

  def _gather_motion_field(
    self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    field = self.fields[field_name]
    clamped = time_steps.clamp(0, field.shape[1] - 1)
    return field[motion_ids.long().unsqueeze(1), clamped.long()]


class _FakeCommandManager:
  def __init__(self, command: _FakeCommand):
    self.command = command

  def get_term(self, name: str) -> _FakeCommand:
    assert name == "motion"
    return self.command


class _FakeScene(dict):
  def __init__(self, *args, env_origins: torch.Tensor | None = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.env_origins = (
      torch.zeros((1, 3), dtype=torch.float32)
      if env_origins is None
      else env_origins
    )


class _SceneWithoutContains:
  def __init__(self, robot: _FakeAsset):
    self._items = {"robot": robot}
    self.env_origins = torch.zeros((1, 3), dtype=torch.float32)

  def __getitem__(self, key):
    if not isinstance(key, str):
      raise KeyError(f"Scene element '{key}' not found")
    return self._items[key]


def _make_env(
  *,
  asset: _FakeAsset,
  command: _FakeCommand | None = None,
  offset: torch.Tensor | None = None,
):
  env = SimpleNamespace()
  env.num_envs = 1
  env.device = "cpu"
  env.physics_dt = 0.02
  env.scene = _FakeScene({"robot": asset})
  if command is not None:
    env.command_manager = _FakeCommandManager(command)
  if offset is None:
    offset = torch.zeros((1, len(asset.joint_names)), dtype=torch.float32)
  env.action_manager = SimpleNamespace(offset=offset)
  env.cfg = SimpleNamespace(robot=SimpleNamespace(mass=10.0))
  return env


def _identity_quat(shape: tuple[int, ...]) -> torch.Tensor:
  quat = torch.zeros((*shape, 4), dtype=torch.float32)
  quat[..., 0] = 1.0
  return quat


def test_target_joint_pos_obs_subtracts_action_offset() -> None:
  joint_count = 3
  fields = {
    "joint_pos": torch.arange(1 * 30 * joint_count, dtype=torch.float32).reshape(
      1, 30, joint_count
    )
    / 10.0
  }
  body_names = ("pelvis",)
  command = _FakeCommand(fields, body_names)
  asset = _FakeAsset(
    data=SimpleNamespace(joint_pos=torch.tensor([[1.0, 2.0, 3.0]])),
    body_names=body_names,
    joint_count=joint_count,
  )
  offset = torch.tensor([[0.1, 0.2, 0.3]])
  env = _make_env(asset=asset, command=command, offset=offset)

  obs = sp.target_joint_pos_obs(env, command_name="motion", horizon="student")

  step_ids = (command.time_steps[:, None] + torch.as_tensor(sp.STUDENT_STEPS)).clamp(
    0, fields["joint_pos"].shape[1] - 1
  )
  expected_target = fields["joint_pos"][command.motion_idx[:, None], step_ids]
  expected_diff = expected_target - (asset.data.joint_pos - offset).unsqueeze(1)
  expected = torch.cat(
    (expected_target.reshape(1, -1), expected_diff.reshape(1, -1)), dim=-1
  )
  assert torch.allclose(obs, expected)


def test_target_pos_b_obs_subtracts_env_origin_from_current_root() -> None:
  body_names = ("pelvis",)
  fields = {
    "body_pos_w": torch.zeros((1, 30, 1, 3), dtype=torch.float32),
    "body_quat_w": _identity_quat((1, 30, 1)),
  }
  fields["body_pos_w"][:, :, 0, :] = torch.tensor([3.0, 0.0, 0.0])
  command = _FakeCommand(fields, body_names)
  asset = _FakeAsset(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[102.0, 0.0, 0.0]]),
      root_link_quat_w=_identity_quat((1,)),
    ),
    body_names=body_names,
    joint_count=1,
  )
  env = _make_env(asset=asset, command=command)
  env.scene.env_origins = torch.tensor([[100.0, 0.0, 0.0]])

  obs = sp.target_pos_b_obs(env, command_name="motion")

  assert torch.allclose(obs[:, :3], torch.tensor([[1.0, 0.0, 0.0]]))


def test_target_feet_contact_fallback_always_returns_two_feet() -> None:
  body_names = ("pelvis", *sp.SP_FEET_BODY_NAMES)
  body_pos_w = torch.zeros((1, 8, 3, 3), dtype=torch.float32)
  body_lin_vel_w = torch.zeros_like(body_pos_w)
  body_pos_w[:, :, body_names.index("left_ankle_roll_link"), 2] = 0.1
  body_pos_w[:, :, body_names.index("right_ankle_roll_link"), 2] = 0.3
  command = _FakeCommand(
    {"body_pos_w": body_pos_w, "body_lin_vel_w": body_lin_vel_w}, body_names
  )
  asset = _FakeAsset(
    data=SimpleNamespace(), body_names=body_names, joint_count=1
  )
  env = _make_env(asset=asset, command=command)

  obs = sp.target_feet_contact_state_obs(env, command_name="motion")

  assert obs.shape == (1, 2)
  assert torch.equal(obs, torch.tensor([[1.0, 0.0]]))


def test_joint_pos_history_subtracts_action_offset_for_current_sample() -> None:
  asset = _FakeAsset(
    data=SimpleNamespace(joint_pos=torch.tensor([[1.0, 2.0, 3.0]])),
    body_names=("pelvis",),
    joint_count=3,
  )
  offset = torch.tensor([[0.1, 0.2, 0.3]])
  env = _make_env(asset=asset, offset=offset)
  term = sp.joint_pos_history(SimpleNamespace(params={"history_steps": (0,)}), env)

  obs = term(env)

  assert torch.allclose(obs, asset.data.joint_pos - offset)


def _make_keypoint_env() -> tuple[SimpleNamespace, _FakeCommand, int]:
  body_names = tuple(sp.SP_REQUIRED_BODY_NAMES)
  body_count = len(body_names)
  keypoint_body_id = body_names.index(sp.SP_KEYPOINT_BODY_NAMES[0])
  frames = 8
  body_pos_w = torch.zeros((1, frames, body_count, 3), dtype=torch.float32)
  body_quat_w = _identity_quat((1, frames, body_count))
  body_lin_vel_w = torch.zeros_like(body_pos_w)
  body_ang_vel_w = torch.zeros_like(body_pos_w)

  body_pos_w[:, :, body_names.index("pelvis"), :] = torch.tensor([0.0, 0.0, 0.0])
  body_pos_w[:, 6, body_names.index("pelvis"), :] = torch.tensor([10.0, 0.0, 0.0])
  body_pos_w[:, 5, keypoint_body_id, :] = torch.tensor([2.0, 0.0, 0.0])
  body_pos_w[:, 6, keypoint_body_id, :] = torch.tensor([11.0, 0.0, 0.0])

  qz90 = torch.tensor(
    [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
    dtype=torch.float32,
  )
  body_quat_w[:, 6, body_names.index("pelvis"), :] = qz90
  body_quat_w[:, 6, keypoint_body_id, :] = qz90

  fields = {
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "body_lin_vel_w": body_lin_vel_w,
    "body_ang_vel_w": body_ang_vel_w,
  }
  command = _FakeCommand(fields, body_names)
  asset_data = SimpleNamespace(
    root_link_pos_w=torch.zeros((1, 3), dtype=torch.float32),
    root_link_quat_w=_identity_quat((1,)),
    root_link_lin_vel_w=torch.zeros((1, 3), dtype=torch.float32),
    root_link_ang_vel_w=torch.zeros((1, 3), dtype=torch.float32),
    body_link_pos_w=torch.zeros((1, body_count, 3), dtype=torch.float32),
    body_link_quat_w=_identity_quat((1, body_count)),
    body_link_lin_vel_w=torch.zeros((1, body_count, 3), dtype=torch.float32),
    body_link_ang_vel_w=torch.zeros((1, body_count, 3), dtype=torch.float32),
  )
  asset = _FakeAsset(data=asset_data, body_names=body_names, joint_count=1)
  return _make_env(asset=asset, command=command), command, keypoint_body_id


def test_target_keypoints_pos_diff_uses_motion_step_zero_root_frame() -> None:
  env, _, _ = _make_keypoint_env()
  term = sp.target_keypoints_pos_b_obs(SimpleNamespace(params={}), env)

  obs = term(env, command_name="motion", required_steps=2, include_diff=True)

  keypoint_count = len(sp.SP_KEYPOINT_BODY_NAMES)
  target_size = 2 * keypoint_count * 3
  target = obs[:, :target_size].reshape(1, 2, keypoint_count, 3)
  diff = obs[:, target_size:].reshape(1, 2, keypoint_count, 3)
  assert torch.allclose(
    target[0, 1, 0], torch.tensor([0.0, -1.0, 0.0]), atol=1.0e-6
  )
  assert torch.allclose(diff[0, 1, 0], torch.tensor([11.0, 0.0, 0.0]))


def test_target_keypoints_rot_diff_uses_motion_step_zero_root_frame() -> None:
  env, _, _ = _make_keypoint_env()
  term = sp.target_keypoints_rot_b_obs(SimpleNamespace(params={}), env)

  obs = term(env, command_name="motion", required_steps=2, include_diff=True)

  keypoint_count = len(sp.SP_KEYPOINT_BODY_NAMES)
  target_size = 2 * keypoint_count * 6
  diff = obs[:, target_size:].reshape(1, 2, keypoint_count, 6)
  qz90 = torch.tensor(
    [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
    dtype=torch.float32,
  )
  expected = sp._rot6d(qz90.reshape(1, 1, 1, 4)).reshape(6)
  assert torch.allclose(diff[0, 1, 0], expected)


def test_projected_gravity_normalizes_helper_output(monkeypatch) -> None:
  def fake_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    del quat
    return vec * 2.0

  monkeypatch.setattr(sp, "_quat_apply_inverse", fake_apply_inverse)

  gravity = sp._projected_gravity(_identity_quat((1,)))

  assert torch.allclose(gravity, torch.tensor([[0.0, 0.0, -1.0]]))
  assert torch.allclose(gravity.norm(dim=-1), torch.ones(1))


def test_feet_contact_state_uses_robot_mass_for_force_normalization() -> None:
  asset = _FakeAsset(
    data=SimpleNamespace(body_mass=torch.tensor([[10.0]], dtype=torch.float32)),
    body_names=("pelvis",),
    joint_count=1,
  )
  env = _make_env(asset=asset)
  force = torch.tensor([[[0.0, 0.0, 98.1]]], dtype=torch.float32)
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      force=force,
      force_history=None,
      current_contact_time=torch.tensor([[0.04]], dtype=torch.float32),
      current_air_time=torch.tensor([[0.0]], dtype=torch.float32),
    )
  )
  env.scene["contact_forces"] = sensor

  obs = sp.feet_contact_state(env, sensor_name="contact_forces")

  assert torch.allclose(obs[:, :3], torch.tensor([[0.0, 0.0, 1.0]]))


def test_feet_contact_state_does_not_require_scene_contains_protocol() -> None:
  asset = _FakeAsset(
    data=SimpleNamespace(body_mass=torch.tensor([[10.0]], dtype=torch.float32)),
    body_names=("pelvis",),
    joint_count=1,
  )
  env = _make_env(asset=asset)
  env.cfg = SimpleNamespace(robot=SimpleNamespace())
  env.scene = _SceneWithoutContains(asset)
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      force=torch.tensor([[[0.0, 0.0, 98.1]]], dtype=torch.float32),
      force_history=None,
      current_contact_time=torch.tensor([[0.04]], dtype=torch.float32),
      current_air_time=torch.tensor([[0.0]], dtype=torch.float32),
    )
  )
  env.scene._items["contact_forces"] = sensor

  obs = sp.feet_contact_state(env, sensor_name="contact_forces")

  assert torch.allclose(obs[:, :3], torch.tensor([[0.0, 0.0, 1.0]]))
