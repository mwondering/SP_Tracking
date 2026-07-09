from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.mdp.multi_commands import MultiMotionCommand


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
