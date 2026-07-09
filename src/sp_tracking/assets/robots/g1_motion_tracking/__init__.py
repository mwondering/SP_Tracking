from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION,
  KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityCfg

from sp_tracking.assets.robots.safety import get_safe_g1_articulation


G1_MOTION_TRACKING_XML = Path(__file__).with_name("g1.xml")


def get_g1_motion_tracking_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(G1_MOTION_TRACKING_XML))


def get_g1_motion_tracking_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_g1_motion_tracking_spec,
    articulation=get_safe_g1_articulation(),
  )
