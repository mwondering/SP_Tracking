from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "src/sp_tracking/assets/robots"
TASK_CFG = (
  ROOT / "src/sp_tracking/conf/task/tracking_bfm_sp_ablation_bfm_actor.yaml"
)


def _object_id(model: mujoco.MjModel, kind, name: str) -> int:
  object_id = mujoco.mj_name2id(model, kind, name)
  assert object_id >= 0, name
  return object_id


def _body_pose(
  model: mujoco.MjModel, data: mujoco.MjData, name: str
) -> tuple[np.ndarray, np.ndarray]:
  body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
  return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def _body_link_velocity(
  model: mujoco.MjModel, data: mujoco.MjData, name: str
) -> tuple[np.ndarray, np.ndarray]:
  """Match mjlab EntityData.body_link_{lin,ang}_vel_w semantics."""
  body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
  root_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  ang_vel = data.cvel[body_id, :3].copy()
  offset = data.subtree_com[root_id] - data.xpos[body_id]
  lin_vel = data.cvel[body_id, 3:].copy() - np.cross(ang_vel, offset)
  return lin_vel, ang_vel


def _set_matching_state(
  sp_model: mujoco.MjModel,
  sp_data: mujoco.MjData,
  bfm_model: mujoco.MjModel,
  bfm_data: mujoco.MjData,
  rng: np.random.Generator,
) -> None:
  for model, data in ((sp_model, sp_data), (bfm_model, bfm_data)):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for joint_id in range(model.njnt):
      if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]
        data.qpos[qpos_adr : qpos_adr + 7] = (0.1, -0.2, 0.8, 1, 0, 0, 0)
        data.qvel[qvel_adr : qvel_adr + 6] = (
          0.2,
          -0.3,
          0.4,
          0.5,
          -0.6,
          0.7,
        )

  for sp_joint_id in range(sp_model.njnt):
    if sp_model.jnt_type[sp_joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    name = mujoco.mj_id2name(
      sp_model, mujoco.mjtObj.mjOBJ_JOINT, sp_joint_id
    )
    bfm_joint_id = _object_id(bfm_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    low = max(
      sp_model.jnt_range[sp_joint_id, 0], bfm_model.jnt_range[bfm_joint_id, 0]
    )
    high = min(
      sp_model.jnt_range[sp_joint_id, 1], bfm_model.jnt_range[bfm_joint_id, 1]
    )
    position = rng.uniform(0.8 * low, 0.8 * high)
    velocity = rng.uniform(-2.0, 2.0)
    sp_data.qpos[sp_model.jnt_qposadr[sp_joint_id]] = position
    bfm_data.qpos[bfm_model.jnt_qposadr[bfm_joint_id]] = position
    sp_data.qvel[sp_model.jnt_dofadr[sp_joint_id]] = velocity
    bfm_data.qvel[bfm_model.jnt_dofadr[bfm_joint_id]] = velocity

  mujoco.mj_forward(sp_model, sp_data)
  mujoco.mj_forward(bfm_model, bfm_data)


def test_bfm_hand_semantic_keypoints_match_sp_xml_pose_and_velocity() -> None:
  cfg = OmegaConf.load(TASK_CFG)
  specs = {item.name: item for item in cfg.obs.semantic_keypoints.heft}
  assert tuple(cfg.reference_views.combined.body_names[-2:]) == (
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
  )

  sp_model = mujoco.MjModel.from_xml_path(
    str(ASSET_ROOT / "g1_sp_tracking/g1.xml")
  )
  bfm_model = mujoco.MjModel.from_xml_path(
    str(ASSET_ROOT / "g1_tracking_bfm/g1.xml")
  )
  sp_data = mujoco.MjData(sp_model)
  bfm_data = mujoco.MjData(bfm_model)
  rng = np.random.default_rng(20260714)

  for _ in range(20):
    _set_matching_state(sp_model, sp_data, bfm_model, bfm_data, rng)
    for side in ("left", "right"):
      spec = specs[f"{side}_hand"]
      hand_pos, hand_rot = _body_pose(
        sp_model, sp_data, f"{side}_hand_mimic"
      )
      hand_lin_vel, hand_ang_vel = _body_link_velocity(
        sp_model, sp_data, f"{side}_hand_mimic"
      )
      parent_pos, parent_rot = _body_pose(bfm_model, bfm_data, spec.body_name)
      parent_lin_vel, parent_ang_vel = _body_link_velocity(
        bfm_model, bfm_data, spec.body_name
      )
      _, correction_rot = _body_pose(
        bfm_model, bfm_data, spec.correction_body_name
      )
      _, correction_ang_vel = _body_link_velocity(
        bfm_model, bfm_data, spec.correction_body_name
      )

      offset = parent_rot @ np.asarray(spec.local_pos, dtype=np.float64)
      correction = correction_rot @ np.asarray(
        spec.correction_local_pos, dtype=np.float64
      )
      semantic_pos = parent_pos + offset + correction
      semantic_lin_vel = (
        parent_lin_vel
        + np.cross(parent_ang_vel, offset)
        + np.cross(correction_ang_vel, correction)
      )

      np.testing.assert_allclose(semantic_pos, hand_pos, atol=1e-12, rtol=0)
      np.testing.assert_allclose(parent_rot, hand_rot, atol=1e-12, rtol=0)
      np.testing.assert_allclose(
        semantic_lin_vel, hand_lin_vel, atol=1e-12, rtol=0
      )
      np.testing.assert_allclose(
        parent_ang_vel, hand_ang_vel, atol=1e-12, rtol=0
      )
