from __future__ import annotations

from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_module

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.mdp.observations import (
  reference_joint_state_window,
)


def _compose(task_name: str):
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=[f"task={task_name}"])


class _CommandManager:
  def __init__(self, command) -> None:
    self.command = command

  def get_term(self, name: str):
    assert name == "motion"
    return self.command


class _WindowCommand:
  def __init__(self) -> None:
    self.time_steps = torch.tensor([10, 20])
    self.motion_idx = torch.tensor([3, 4])
    self.calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []

  def _gather_motion_field(self, field_name, motion_idx, time_steps):
    self.calls.append((field_name, motion_idx.clone(), time_steps.clone()))
    value = 1.0 if field_name == "joint_pos" else 2.0
    return torch.full((*time_steps.shape, 29), value)


def test_reference_joint_state_window_matches_source_580_layout() -> None:
  command = _WindowCommand()
  env = SimpleNamespace(num_envs=2, command_manager=_CommandManager(command))

  result = reference_joint_state_window(
    env, "motion", history_steps=5, future_steps=5
  )

  assert result.shape == (2, 580)
  assert torch.all(result[:, :290] == 1.0)
  assert torch.all(result[:, 290:] == 2.0)
  expected_steps = torch.tensor(
    [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
  )
  assert [call[0] for call in command.calls] == ["joint_pos", "joint_vel"]
  assert all(torch.equal(call[2], expected_steps) for call in command.calls)


def test_wbteleop_baseline_is_808_actor_with_original_bfm_critic() -> None:
  task_name = "tracking_bfm_wbteleop_actor_bfm_critic"
  cfg = _compose(task_name)
  bfm = prepare_train_cfg(_compose("tracking_bfm"))
  prepared = prepare_train_cfg(cfg)
  env = build_env_cfg(cfg.task)

  actor_terms = env.observations["actor"].terms
  assert tuple(actor_terms) == (
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
  )
  assert actor_terms["command"].func is reference_joint_state_window
  assert actor_terms["command"].params == {
    "command_name": "motion",
    "history_steps": 5,
    "future_steps": 5,
  }
  # 580 reference q/qdot + 3 anchor pos + 6 anchor rot + 14*(3+6)
  # body pose + 2*3 base velocity + 3*29 joint/action state.
  assert 580 + 3 + 6 + 14 * 9 + 6 + 3 * 29 == 808

  command = env.commands["motion"]
  assert command.history_steps == 0
  assert command.future_steps == 1
  assert prepared.agent.obs_groups == {
    "actor": ("actor",),
    "critic": ("critic",),
  }
  assert prepared.agent.critic == bfm.agent.critic
  assert prepared.agent.actor == bfm.agent.actor
  assert tuple(prepared.env.observations) == ("actor", "critic")

