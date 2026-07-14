from __future__ import annotations

from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_module

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.mdp.observations import (
  motion_ref_ang_vel,
  ref_limb_ee_pose_b,
  reference_joint_state_window,
  robot_limb_ee_pose_b,
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


def test_wbteleop_baseline_is_deployable_886_actor_with_original_bfm_critic() -> None:
  task_name = "tracking_bfm_wbteleop_actor_bfm_critic"
  cfg = _compose(task_name)
  bfm = prepare_train_cfg(_compose("tracking_bfm"))
  prepared = prepare_train_cfg(cfg)
  env = build_env_cfg(cfg.task)

  actor_terms = env.observations["actor"].terms
  assert tuple(actor_terms) == (
    "command",
    "ref_limb_ee_pose_b",
    "motion_ref_ang_vel",
    "robot_limb_ee_pose_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  assert actor_terms["ref_limb_ee_pose_b"].func is ref_limb_ee_pose_b
  assert actor_terms["robot_limb_ee_pose_b"].func is robot_limb_ee_pose_b
  assert actor_terms["motion_ref_ang_vel"].func is motion_ref_ang_vel
  assert actor_terms["ref_limb_ee_pose_b"].params["anchor_body_name"] == "pelvis"
  assert actor_terms["robot_limb_ee_pose_b"].params["anchor_body_name"] == "pelvis"
  for term_name in (
    "ref_limb_ee_pose_b",
    "robot_limb_ee_pose_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  ):
    assert actor_terms[term_name].history_length == 5
  assert "base_lin_vel" not in actor_terms
  assert "motion_anchor_pos_b" not in actor_terms
  assert "motion_anchor_ori_b" not in actor_terms

  # 58 current reference q/qdot + two 5-frame, four-limb SE(3) views
  # + reference angular velocity + five-frame deployable proprioception.
  assert 58 + 2 * (5 * 4 * 9) + 3 + 5 * (3 + 3 + 3 * 29) == 886

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
