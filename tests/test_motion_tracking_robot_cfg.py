import pytest

from sp_tracking.assets.robots.g1_motion_tracking import (
  G1_MOTION_TRACKING_ARTICULATION,
  get_g1_motion_tracking_robot_cfg,
)
from sp_tracking.assets.robots.g1_tracking_bfm import (
  get_g1_tracking_bfm_robot_cfg,
)


def _actuator_for(pattern: str):
  for actuator in G1_MOTION_TRACKING_ARTICULATION.actuators:
    if pattern in actuator.target_names_expr:
      return actuator
  raise AssertionError(f"No actuator matches {pattern}")


def test_motion_tracking_asset_uses_reference_actuator_parameters() -> None:
  hip_pitch = _actuator_for(".*_hip_pitch_joint")
  wrist_pitch = _actuator_for(".*_wrist_pitch_joint")

  assert hip_pitch.armature == pytest.approx(0.025101925)
  assert hip_pitch.stiffness == pytest.approx(99.09842777666113)
  assert hip_pitch.damping == pytest.approx(6.3088018534966395)
  assert hip_pitch.effort_limit == pytest.approx(139.0)
  assert wrist_pitch.armature == pytest.approx(0.0021812)
  assert wrist_pitch.stiffness == pytest.approx(8.611032447370201)
  assert wrist_pitch.damping == pytest.approx(0.548195351665136)
  assert wrist_pitch.effort_limit == pytest.approx(13.4)


def test_motion_tracking_articulation_is_isolated_from_tracking_bfm() -> None:
  sp_robot = get_g1_motion_tracking_robot_cfg()
  baseline_robot = get_g1_tracking_bfm_robot_cfg()

  assert sp_robot.articulation is G1_MOTION_TRACKING_ARTICULATION
  assert baseline_robot.articulation is not G1_MOTION_TRACKING_ARTICULATION
