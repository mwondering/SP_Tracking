import torch

from sp_tracking.tasks.tracking.mdp.actions import MotionTrackingJointPositionAction


def test_motion_tracking_action_torque_schedule_never_exceeds_default_force_range() -> None:
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

  assert applied_scale == 1.0
  assert torch.equal(model.actuator_forcerange[0], action._default_forcerange)
