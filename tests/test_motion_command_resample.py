from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.mdp.multi_commands import (
  MultiMotionCommand,
  apply_reset_ground_clearance,
  clamp_reset_joint_velocity,
)


class _FakeSim:
  def __init__(self) -> None:
    self.forward_calls = 0

  def forward(self) -> None:
    self.forward_calls += 1


class _ResampleForwardCommand(MultiMotionCommand):
  def __init__(self) -> None:
    pass

  @property
  def command(self) -> torch.Tensor:
    return torch.empty(1, 0)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return torch.tensor([[[1.0, 0.0, 0.0]]]).squeeze(1)

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]])

  @property
  def body_pos_w(self) -> torch.Tensor:
    return torch.tensor([[[2.0, 0.0, 0.0]]])

  @property
  def body_quat_w(self) -> torch.Tensor:
    return torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    if self._env.sim.forward_calls == 0:
      return torch.tensor([[0.0, 0.0, 0.0]])
    return torch.tensor([[10.0, 0.0, 0.0]])

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]])

  def _update_metrics(self) -> None:
    return None

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.resampled_env_ids = env_ids.clone()


def test_update_command_forwards_sim_after_mid_episode_resample() -> None:
  command = _ResampleForwardCommand()
  command.cfg = SimpleNamespace(sampling_mode="uniform", body_names=("root",))
  command._env = SimpleNamespace(sim=_FakeSim())
  command.time_steps = torch.tensor([0])
  command.motion_length = torch.tensor([1])

  command._update_command()

  assert command._env.sim.forward_calls == 1
  assert command.resampled_env_ids.tolist() == [0]
  assert torch.allclose(command.body_pos_relative_w, torch.tensor([[[11.0, 0.0, 0.0]]]))


def test_reset_ground_clearance_accounts_for_lift_noise_and_rotation() -> None:
  root_pos = torch.tensor([[0.0, 0.0, 0.05], [1.0, 2.0, 1.05]])
  body_pos_w = torch.tensor(
    [
      [[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]],
      [[1.0, 2.0, 1.05], [1.0, 2.0, 0.95]],
    ]
  )
  env_origins = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
  position_noise = torch.tensor([[0.1, -0.2, 0.0], [-0.1, 0.2, 0.0]])
  identity_quat = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )

  adjusted = apply_reset_ground_clearance(
    root_pos,
    body_pos_w,
    env_origins,
    position_noise,
    identity_quat,
    root_lift_height=0.04,
    min_body_z=0.0,
  )

  relative_body_z = body_pos_w[..., 2] - root_pos[:, None, 2]
  predicted_min_z = adjusted[:, None, 2] + relative_body_z
  assert torch.allclose(
    adjusted, torch.tensor([[0.1, -0.2, 0.1], [0.9, 2.2, 1.1]])
  )
  assert torch.allclose(
    predicted_min_z.amin(dim=1), env_origins[:, 2]
  )


def test_reset_ground_clearance_is_disabled_by_default() -> None:
  root_pos = torch.tensor([[1.0, 2.0, 3.0]])
  position_noise = torch.tensor([[0.1, 0.2, -0.3]])

  adjusted = apply_reset_ground_clearance(
    root_pos,
    body_pos_w=root_pos.unsqueeze(1),
    env_origins=torch.zeros((1, 3)),
    position_noise=position_noise,
    orientation_delta=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    root_lift_height=0.0,
    min_body_z=None,
  )

  assert torch.equal(adjusted, root_pos + position_noise)


def test_reset_joint_velocity_clamp_is_opt_in() -> None:
  joint_vel = torch.tensor([[-15.0, -5.0, 8.0, 20.0]])

  assert torch.equal(clamp_reset_joint_velocity(joint_vel, None), joint_vel)
  assert torch.equal(
    clamp_reset_joint_velocity(joint_vel, 10.0),
    torch.tensor([[-10.0, -5.0, 8.0, 10.0]]),
  )
