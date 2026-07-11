from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION,
  KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


G1_MOTION_TRACKING_XML = Path(__file__).with_name("g1.xml")


# Keep this articulation local to the SP asset.  These are the exact actuator
# parameters used by UNICTL/motion_tracking's G1_ARTICULATION; the regular
# tracking_bfm asset continues to use mjlab's articulation unchanged.
G1_MOTION_TRACKING_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPositionActuatorCfg(
      target_names_expr=(
        ".*_elbow_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_wrist_roll_joint",
      ),
      armature=0.003609725,
      stiffness=14.25062309787429,
      damping=0.907222843292423,
      effort_limit=25.0,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(
        ".*_hip_pitch_joint",
        ".*_hip_roll_joint",
        ".*_knee_joint",
      ),
      armature=0.025101925,
      stiffness=99.09842777666113,
      damping=6.3088018534966395,
      effort_limit=139.0,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
      armature=0.010177520,
      stiffness=40.17923847137318,
      damping=2.5578897650279457,
      effort_limit=88.0,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
      armature=0.0021812,
      stiffness=8.611032447370201,
      damping=0.548195351665136,
      effort_limit=13.4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
      armature=0.00721945,
      stiffness=28.50124619574858,
      damping=1.814445686584846,
      effort_limit=35.0,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
      armature=0.00721945,
      stiffness=28.50124619574858,
      damping=1.814445686584846,
      effort_limit=35.0,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_motion_tracking_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(G1_MOTION_TRACKING_XML))


def get_g1_motion_tracking_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_g1_motion_tracking_spec,
    articulation=G1_MOTION_TRACKING_ARTICULATION,
  )
