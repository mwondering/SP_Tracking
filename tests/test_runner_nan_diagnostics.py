from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.rl.runner import (
  _format_nonfinite_action_diagnostics,
  _format_nonfinite_env_output_diagnostics,
)


class _CommandManager:
  def __init__(self, command):
    self.command = command

  def get_term(self, name: str):
    assert name == "motion"
    return self.command


def test_nan_diagnostics_identify_concatenated_obs_term_and_motion_file() -> None:
  obs = {
    "actor": torch.tensor(
      [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, float("nan"), 5.0],
      ]
    )
  }
  command = SimpleNamespace(
    motion_idx=torch.tensor([0, 3]),
    time_steps=torch.tensor([11, 17]),
    motion_store=SimpleNamespace(
      motion_files=["m0.npz", "m1.npz", "m2.npz", "bad_motion.npz"]
    ),
  )
  unwrapped = SimpleNamespace(
    observation_manager=SimpleNamespace(
      active_terms={"actor": ["good", "bad"]},
      group_obs_term_dim={"actor": [(2,), (3,)]},
      group_obs_concatenate={"actor": True},
    ),
    command_manager=_CommandManager(command),
  )
  env = SimpleNamespace(unwrapped=unwrapped)

  message = _format_nonfinite_env_output_diagnostics(
    env,
    obs,
    rewards=torch.zeros(2),
    dones=torch.zeros(2, dtype=torch.long),
  )

  assert "obs[actor]/bad" in message
  assert "envs=[1]" in message
  assert "motion_ids=[3]" in message
  assert "time_steps=[17]" in message
  assert "bad_motion.npz" in message


def test_action_diagnostics_identify_nonfinite_policy_actions() -> None:
  command = SimpleNamespace(
    motion_idx=torch.tensor([2, 4]),
    time_steps=torch.tensor([5, 9]),
    motion_store=SimpleNamespace(
      motion_files=["m0.npz", "m1.npz", "source.npz", "m3.npz", "nan_action.npz"]
    ),
  )
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(command_manager=_CommandManager(command))
  )

  message = _format_nonfinite_action_diagnostics(
    env,
    torch.tensor([[0.0, 1.0], [float("nan"), 0.5]]),
  )

  assert "Policy action contains NaN/Inf" in message
  assert "envs=[1]" in message
  assert "motion_ids=[4]" in message
  assert "time_steps=[9]" in message
  assert "nan_action.npz" in message
