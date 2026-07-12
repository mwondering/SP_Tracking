from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.mdp.multi_commands import (
  MultiMotionCommand,
  RewindCfg,
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


def test_rewind_sampling_only_rewinds_non_timeout_failures(monkeypatch) -> None:
  command = MultiMotionCommand.__new__(MultiMotionCommand)
  command.cfg = SimpleNamespace(
    rewind=RewindCfg(enabled=True, failure_probability=1.0),
    if_log_metrics=False,
  )
  command._env = SimpleNamespace(
    device="cpu",
    num_envs=3,
    termination_manager=SimpleNamespace(
      terminated=torch.tensor([True, True, False]),
      time_outs=torch.tensor([False, True, False]),
    )
  )
  command.time_steps = torch.tensor([19, 24, 31])
  command.motion_idx = torch.tensor([0, 0, 0])
  command.bin_width_steps = 10
  command.metrics = {}
  command._invalidate_reference_cache = lambda: None
  command._stage_pre_resample_adaptive_stats = lambda env_ids: None
  command._compute_motion_bin_indices = lambda time_steps, motion_idx: torch.div(
    time_steps, 10, rounding_mode="floor"
  )
  monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(*args, **kwargs))

  remaining = command._prepare_reset_sampling(torch.tensor([0, 1, 2]))

  # env 0 is a failure and rewinds to the first frame of bin 1.  env 1 is a
  # timeout and therefore stays on the normal adaptive-reset path.
  assert command.time_steps.tolist() == [10, 24, 31]
  assert remaining.tolist() == [1, 2]


def test_sp_tracking_reset_clears_consecutive_termination_buffers() -> None:
  command = MultiMotionCommand.__new__(MultiMotionCommand)
  command.cfg = SimpleNamespace(boot_indicator_max=25, sliding_root_xy_reward=False)
  command.boot_indicator = torch.zeros((2, 1))
  command.feet_standing = torch.ones((2, 2), dtype=torch.bool)
  command._body_z_termination_buffer = torch.tensor([3, 5], dtype=torch.int32)
  command._gravity_dir_termination_buffer = torch.tensor([2, 5], dtype=torch.int32)
  command.reward_root_ref_xy_history_w = torch.zeros((2, 1, 2))
  command.reward_root_actual_xy_history_w = torch.zeros((2, 1, 2))
  command._reward_root_history_slot = torch.zeros(2, dtype=torch.long)
  command.reward_root_pos_w = torch.zeros((2, 3))
  command.reward_root_quat_w = torch.zeros((2, 4))
  command._reference_root_pos_w = lambda: torch.tensor(
    [[0.0, 0.0, 1.0], [1.0, 2.0, 1.0]]
  )
  command._reference_root_quat_w = lambda: torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )

  command._reset_sp_tracking_state(
    torch.tensor([1]), actual_root_pos_w=torch.tensor([[3.0, 4.0, 1.0]])
  )

  assert command._body_z_termination_buffer.tolist() == [3, 0]
  assert command._gravity_dir_termination_buffer.tolist() == [2, 0]
  assert command.boot_indicator[:, 0].tolist() == [0.0, 25.0]
