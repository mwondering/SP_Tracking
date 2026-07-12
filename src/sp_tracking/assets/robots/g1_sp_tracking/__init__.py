from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


G1_SP_TRACKING_XML = Path(__file__).with_name("g1.xml")


# Keep the wrapper semantics aligned with active_adaptation/assets/g1.py.  The
# XML itself is shared byte-for-byte with the reference repository; these are
# the configuration-level details that are not encoded in the XML.
G1_FULL_COLLISION_ALL_PRIORITY = replace(
  FULL_COLLISION,
  condim=3,
  priority=1,
)

G1_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.74),
  joint_pos={
    ".*_hip_pitch_joint": -0.28,
    ".*_knee_joint": 0.5,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
    ".*_wrist_roll_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
    ".*": 0.0,
  },
  joint_vel={".*": 0.0},
)

# This is the canonical policy/data ordering used by SP tracking.  MuJoCo
# stores G1 joints in XML tree order, so downstream SP terms explicitly use
# this order when building action and observation tensors.
G1_SP_JOINT_ORDER = (
  "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
  "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
  "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
  "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
  "right_ankle_pitch_joint", "left_shoulder_roll_joint",
  "right_shoulder_roll_joint", "left_ankle_roll_joint",
  "right_ankle_roll_joint", "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
  "left_wrist_roll_joint", "right_wrist_roll_joint",
  "left_wrist_pitch_joint", "right_wrist_pitch_joint",
  "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)


def _swap_left_right(name: str) -> str:
  if name.startswith("left_"):
    return "right_" + name[len("left_") :]
  if name.startswith("right_"):
    return "left_" + name[len("right_") :]
  return name


def _joint_symmetry_sign(name: str, axis: tuple[float, float, float]) -> int:
  lower = name.lower()
  if "roll" in lower or "yaw" in lower or "arm_yaw" in lower:
    return -1
  if "pitch" in lower or "knee" in lower or "elbow" in lower:
    return 1
  dominant_axis = max(range(3), key=lambda index: abs(axis[index]))
  return 1 if dominant_axis == 1 else -1


def _build_xml_symmetry_maps() -> tuple[dict[str, tuple[int, str]], dict[str, str]]:
  """Reproduce the reference task's deterministic auto-symmetry mapping.

  The upstream helper additionally validates the generated mapping.  mjlab's
  current EntityCfg does not consume these mappings itself, but retaining them
  on the SP asset keeps config consumers and future symmetry ablations aligned
  with the reference wrapper without importing the legacy project at runtime.
  """
  root = ET.parse(G1_SP_TRACKING_XML).getroot()
  joint_axes: dict[str, tuple[float, float, float]] = {}
  body_names: list[str] = []
  for body in root.findall(".//body"):
    body_name = body.get("name")
    if body_name is not None:
      body_names.append(body_name)
    for joint in body.findall("joint"):
      joint_name = joint.get("name")
      if joint_name is None:
        continue
      joint_axes[joint_name] = tuple(
        float(value) for value in joint.get("axis", "0 0 0").split()
      )

  joint_map: dict[str, tuple[int, str]] = {}
  for name, axis in joint_axes.items():
    mirrored = _swap_left_right(name)
    if mirrored not in joint_axes:
      mirrored = name
    sign = _joint_symmetry_sign(name, axis)
    if mirrored != name:
      mirrored_axis = joint_axes[mirrored]
      if sum(a * b for a, b in zip(axis, mirrored_axis, strict=True)) < 0.0:
        sign = -sign
    joint_map[name] = (sign, mirrored)

  body_map = {
    name: (_swap_left_right(name) if _swap_left_right(name) in body_names else name)
    for name in body_names
  }
  return joint_map, body_map


G1_SP_JOINT_SYMMETRY_MAP, G1_SP_SPATIAL_SYMMETRY_MAP = _build_xml_symmetry_maps()


# Keep this articulation local to the SP asset.  These are the exact actuator
# parameters used by the reference G1 articulation; the regular
# tracking_bfm asset continues to use mjlab's articulation unchanged.
G1_SP_TRACKING_ARTICULATION = EntityArticulationInfoCfg(
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


def get_g1_sp_tracking_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(G1_SP_TRACKING_XML))


def get_g1_sp_tracking_robot_cfg() -> EntityCfg:
  cfg = EntityCfg(
    init_state=G1_INIT_STATE,
    collisions=(G1_FULL_COLLISION_ALL_PRIORITY,),
    spec_fn=get_g1_sp_tracking_spec,
    articulation=G1_SP_TRACKING_ARTICULATION,
  )
  # These attributes are intentionally attached for compatibility with the
  # original wrapper.  EntityCfg is not slotted, and mjlab preserves them on
  # the asset configuration for downstream consumers.
  cfg.joint_symmetry_mapping = G1_SP_JOINT_SYMMETRY_MAP
  cfg.spatial_symmetry_mapping = G1_SP_SPATIAL_SYMMETRY_MAP
  cfg.joint_name_order = G1_SP_JOINT_ORDER
  return cfg
