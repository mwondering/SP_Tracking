from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.mdp.actions import MotionTrackingJointPositionAction


def test_motion_tracking_action_torque_schedule_applies_configured_scale() -> None:
  action = object.__new__(MotionTrackingJointPositionAction)
  action.cfg = type(
    "Cfg",
    (),
    {
      "torque_limit_progress_range": (0.0, 1.0),
      "torque_limit_scale_range": (4.0, 2.0),
    },
  )()
  action._torque_limit_scale = None
  action._default_forcerange = torch.tensor([[-35.0, 35.0], [-5.0, 5.0]])
  action._ctrl_ids = torch.tensor([0, 1])
  model = type(
    "Model",
    (),
    {"actuator_forcerange": torch.zeros((1, 2, 2), dtype=torch.float32)},
  )()
  action._env = type("Env", (), {"sim": type("Sim", (), {"model": model})()})()

  applied_scale = action._schedule_torque_limit(0.0)

  assert applied_scale == 4.0
  assert torch.equal(
    model.actuator_forcerange[0], action._default_forcerange * 4.0
  )


def test_motion_tracking_action_clamps_raw_policy_action() -> None:
  action = object.__new__(MotionTrackingJointPositionAction)
  action.cfg = SimpleNamespace(raw_action_clip=10.0)
  action._raw_actions = torch.zeros((1, 2))
  action._action_history = torch.zeros((1, 3, 2))
  action._substep = 7

  action.process_actions(torch.tensor([[12.0, -15.0]]))

  assert torch.equal(action._raw_actions, torch.tensor([[10.0, -10.0]]))
  assert torch.equal(action._action_history[:, 0], action._raw_actions)
  assert action._substep == 0


def test_motion_tracking_full_curriculum_matches_student_finetune_mode() -> None:
  action = object.__new__(MotionTrackingJointPositionAction)
  action.cfg = SimpleNamespace(curriculum_mode="full")
  action.max_delay = 2
  action.delay_probs = torch.zeros(3)
  action._schedule_torque_limit = lambda progress: progress

  state = action.step_schedule(progress=0.0)

  assert torch.allclose(action.delay_probs, torch.full((3,), 1.0 / 3.0))
  assert state["torque_limit_scale"] == 1.0


def test_motion_tracking_action_holds_boot_target_for_two_substeps() -> None:
  action = object.__new__(MotionTrackingJointPositionAction)
  action.cfg = SimpleNamespace(boot_delay_steps=2, clip=None)
  action._env = SimpleNamespace(num_envs=1, device="cpu")
  action.delay = torch.zeros((1, 1), dtype=torch.long)
  action._decimation = 4
  action._history_len = 2
  action._action_history = torch.full((1, 2, 2), 3.0)
  action.applied_action = torch.zeros((1, 2))
  action.alpha = torch.ones((1, 1))
  action._scale = 1.0
  action._default_offset = torch.zeros((1, 2))
  action.joint_offset = torch.zeros((1, 2))
  action.boot_delay = torch.zeros((1, 1), dtype=torch.long)
  action.boot_target = torch.zeros((1, 2))
  env_ids = torch.tensor([0])

  action.set_boot_target(env_ids, torch.tensor([[1.0, 2.0]]))
  action._update_processed_actions(0)
  first = action._processed_actions.clone()
  action._update_processed_actions(1)
  second = action._processed_actions.clone()
  action._update_processed_actions(2)
  third = action._processed_actions.clone()

  assert torch.equal(first, torch.tensor([[1.0, 2.0]]))
  assert torch.equal(second, torch.tensor([[1.0, 2.0]]))
  assert torch.equal(third, torch.tensor([[3.0, 3.0]]))
