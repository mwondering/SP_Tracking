from types import SimpleNamespace

from sp_tracking.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner


class _ScheduleTerm:
  def __init__(self):
    self.calls = []

  def step_schedule(self, progress: float, iteration: int):
    self.calls.append((progress, iteration))
    return {"seen": progress}


class _ActionManager:
  active_terms = ["joint_pos"]

  def __init__(self, term):
    self.term = term

  def get_term(self, name: str):
    assert name == "joint_pos"
    return self.term


class _EventManager:
  active_terms = {"step": ["perturb_body_wrench"]}

  def __init__(self, term):
    self.term = term

  def get_term_cfg(self, name: str):
    assert name == "perturb_body_wrench"
    return SimpleNamespace(func=self.term)


class _CurriculumManager:
  active_terms = ["motion_tracking_progress"]


def test_runner_steps_motion_tracking_curriculum_terms() -> None:
  action_term = _ScheduleTerm()
  event_term = _ScheduleTerm()
  unwrapped = SimpleNamespace(
    action_manager=_ActionManager(action_term),
    event_manager=_EventManager(event_term),
    curriculum_manager=_CurriculumManager(),
  )
  runner = object.__new__(MotionTrackingOnPolicyRunner)
  runner.env = SimpleNamespace(unwrapped=unwrapped)
  runner.cfg = {"max_iterations": 100}

  runner._step_motion_tracking_curriculum(iteration=24)

  assert action_term.calls == [(0.25, 24)]
  assert event_term.calls == [(0.25, 24)]
  assert unwrapped._motion_tracking_curriculum_state["progress"] == 0.25
