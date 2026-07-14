from types import SimpleNamespace

from sp_tracking.assets.robots.g1_sp_tracking import (
  G1_SP_JOINT_ORDER,
  get_g1_sp_tracking_robot_cfg,
)
from sp_tracking.tasks.tracking.rl import symmetry


class _EventManager:
  def __init__(self, funcs):
    self.funcs = funcs

  def get_term_cfg(self, name: str):
    return SimpleNamespace(func=self.funcs[name])


def test_heft_symmetry_uses_actual_dr_observation_column_order() -> None:
  robot_cfg = get_g1_sp_tracking_robot_cfg()
  xml_order = tuple(reversed(G1_SP_JOINT_ORDER))
  motor = SimpleNamespace(
    kp_names=xml_order,
    kd_names=xml_order,
    arm_names=xml_order,
    fric_names=(),
  )
  joint_offset = SimpleNamespace(joint_names=xml_order)
  root = SimpleNamespace(
    scene={"robot": SimpleNamespace(cfg=robot_cfg)},
    event_manager=_EventManager(
      {
        "motor_params_implicit": motor,
        "random_joint_offset": joint_offset,
      }
    ),
  )
  env = SimpleNamespace(unwrapped=root)

  transforms, action = symmetry._build(env)

  assert transforms["policy"].perm.numel() == 1729
  assert transforms["priv"].perm.numel() == 4622
  assert transforms["priv_critic"].perm.numel() == 122
  assert action.perm.numel() == 29
  offset = 29 * 3 + 3
  left_column = xml_order.index("left_hip_pitch_joint")
  right_column = xml_order.index("right_hip_pitch_joint")
  assert transforms["priv_critic"].perm[offset + left_column].item() == (
    offset + right_column
  )
