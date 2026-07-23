from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.mdp import observations as tracking_observations
from sp_tracking.tasks.tracking.mdp import rewards as tracking_rewards
from sp_tracking.tasks.tracking.mdp import terminations as tracking_terminations


class _CommandManager:
  def __init__(self, command) -> None:
    self.command = command

  def get_term(self, name: str):
    assert name == "motion"
    return self.command


def _identity_quaternions(num_bodies: int) -> torch.Tensor:
  quats = torch.zeros((1, num_bodies, 4))
  quats[..., 0] = 1.0
  return quats


def test_old_tracking_view_selects_its_body_set_from_union_reference() -> None:
  # SP-reward and old-observation body sets can coexist in one command cache.
  # The old view must still emit exactly 3/6 values per requested body.
  body_names = (
    "pelvis",
    "sp_only_body",
    "torso_link",
    "legacy_left",
    "legacy_right",
  )
  body_pos = torch.tensor(
    [[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [1.0, 0.0, 0.0],
      [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
  )
  robot_pos = body_pos + torch.tensor([[[0.0, 0.0, 0.0]]])
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=body_names),
    body_pos_w=body_pos,
    body_quat_w=_identity_quaternions(len(body_names)),
    robot_body_pos_w=robot_pos,
    robot_body_quat_w=_identity_quaternions(len(body_names)),
  )
  env = SimpleNamespace(num_envs=1, command_manager=_CommandManager(command))
  old_body_names = ("pelvis", "torso_link", "legacy_left", "legacy_right")

  anchor = tracking_observations.motion_anchor_pos_b(
    env, "motion", anchor_body_name="torso_link"
  )
  body_pos_obs = tracking_observations.robot_body_pos_b(
    env,
    "motion",
    body_names=old_body_names,
    anchor_body_name="torso_link",
  )
  body_ori_obs = tracking_observations.robot_body_ori_b(
    env,
    "motion",
    body_names=old_body_names,
    anchor_body_name="torso_link",
  )

  assert anchor.shape == (1, 3)
  assert body_pos_obs.shape == (1, 3 * len(old_body_names))
  assert body_ori_obs.shape == (1, 6 * len(old_body_names))
  # The SP-only body is intentionally excluded from the legacy view.
  assert torch.allclose(body_pos_obs[0, :3], torch.tensor([-1.0, 0.0, 0.0]))

  # Legacy reward/termination terms use the same torso view, but preserve the
  # command's yaw-aligned relative-pose construction rather than raw indexing.
  assert torch.allclose(
    tracking_rewards.motion_relative_body_position_error_exp(
      env,
      "motion",
      std=1.0,
      body_names=old_body_names,
      anchor_body_name="torso_link",
    ),
    torch.ones(1),
  )
  assert torch.allclose(
    tracking_rewards.motion_relative_body_orientation_error_exp(
      env,
      "motion",
      std=1.0,
      body_names=old_body_names,
      anchor_body_name="torso_link",
    ),
    torch.ones(1),
  )
  assert not tracking_terminations.bad_motion_body_pos_z_only(
    env,
    "motion",
    threshold=0.01,
    body_names=old_body_names,
    anchor_body_name="torso_link",
  ).item()


def test_global_body_position_termination_checks_each_selected_link() -> None:
  body_names = (
    "pelvis",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  reference_pos = torch.zeros((4, len(body_names), 3))
  robot_pos = reference_pos.clone()
  robot_pos[0, body_names.index("left_wrist_yaw_link"), 0] = 0.5
  robot_pos[1, body_names.index("right_wrist_yaw_link"), 0] = 0.501
  robot_pos[2, body_names.index("left_ankle_roll_link"), :2] = torch.tensor(
    [0.3, 0.39]
  )
  robot_pos[3, body_names.index("pelvis"), 2] = -0.51
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=body_names),
    body_pos_w=reference_pos,
    robot_body_pos_w=robot_pos,
  )
  env = SimpleNamespace(num_envs=4, command_manager=_CommandManager(command))

  outputs = [
    tracking_terminations.bad_motion_body_pos_global(
      env,
      "motion",
      threshold=0.5,
      body_names=body_names,
      consecutive_steps=5,
    )
    for _ in range(5)
  ]

  assert all(not output.any() for output in outputs[:4])
  assert torch.equal(outputs[4], torch.tensor([False, True, False, True]))


def test_global_body_position_termination_buffer_resets_after_recovery() -> None:
  body_names = ("pelvis",)
  reference_pos = torch.zeros((1, 1, 3))
  robot_pos = torch.tensor([[[0.51, 0.0, 0.0]]])
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=body_names),
    body_pos_w=reference_pos,
    robot_body_pos_w=robot_pos,
  )
  env = SimpleNamespace(num_envs=1, command_manager=_CommandManager(command))

  def terminate() -> torch.Tensor:
    return tracking_terminations.bad_motion_body_pos_global(
      env,
      "motion",
      threshold=0.5,
      body_names=body_names,
      consecutive_steps=5,
    )

  assert not terminate().item()
  assert not terminate().item()
  command.robot_body_pos_w.zero_()
  assert not terminate().item()
  command.robot_body_pos_w[..., 0] = 0.51
  assert [terminate().item() for _ in range(4)] == [False] * 4
  assert terminate().item()
