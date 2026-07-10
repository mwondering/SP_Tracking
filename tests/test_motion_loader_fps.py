from pathlib import Path

import numpy as np
import pytest
import torch

from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  LargeDatasetMotionStore,
)
from sp_tracking.tasks.tracking.mdp.motion_fk import MotionFKHelper
from sp_tracking.tasks.tracking.mdp.multi_commands import (
  _MUJOCO_JOINT_NAMES,
  MotionLoader,
  MultiMotionLoader,
)


SP_BODY_NAMES = (
  "pelvis",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "head_mimic",
  "left_shoulder_yaw_link",
  "left_wrist_roll_link",
  "left_hand_mimic",
  "right_shoulder_yaw_link",
  "right_wrist_roll_link",
  "right_hand_mimic",
)
SP_BODY_INDEXES = torch.tensor(
  [0, 3, 4, 6, 10, 11, 13, 17, 18, 21, 23, 26, 29, 31, 34],
  dtype=torch.long,
)


def _write_motion(path: Path, *, include_fps: bool) -> None:
  frames = 4
  joints = 3
  bodies = 3
  motion = {
    "joint_pos": np.zeros((frames, joints), dtype=np.float32),
    "joint_vel": np.zeros((frames, joints), dtype=np.float32),
    "body_pos_w": np.zeros((frames, bodies, 3), dtype=np.float32),
    "body_quat_w": np.zeros((frames, bodies, 4), dtype=np.float32),
    "body_lin_vel_w": np.zeros((frames, bodies, 3), dtype=np.float32),
    "body_ang_vel_w": np.zeros((frames, bodies, 3), dtype=np.float32),
  }
  motion["body_quat_w"][..., 0] = 1.0
  if include_fps:
    motion["fps"] = np.asarray([60.0], dtype=np.float32)
  np.savez(path, **motion)


def _write_legacy_g1_motion(path: Path) -> None:
  frames = 6
  joints = 29
  legacy_bodies = 30
  motion = {
    "joint_pos": np.zeros((frames, joints), dtype=np.float32),
    "joint_vel": np.zeros((frames, joints), dtype=np.float32),
    "body_pos_w": np.zeros((frames, legacy_bodies, 3), dtype=np.float32),
    "body_quat_w": np.zeros((frames, legacy_bodies, 4), dtype=np.float32),
    "body_lin_vel_w": np.zeros((frames, legacy_bodies, 3), dtype=np.float32),
    "body_ang_vel_w": np.zeros((frames, legacy_bodies, 3), dtype=np.float32),
    "fps": np.asarray([50.0], dtype=np.float32),
  }
  motion["body_pos_w"][:, 0, 2] = 0.75
  motion["body_quat_w"][..., 0] = 1.0
  np.savez(path, **motion)


def _sp_fk_helper() -> MotionFKHelper:
  return MotionFKHelper.from_mjcf_path(
    xml_path="src/sp_tracking/assets/robots/g1_motion_tracking/g1.xml",
    dataset_joint_names=_MUJOCO_JOINT_NAMES,
    output_body_names=SP_BODY_NAMES,
    device="cpu",
  )


def test_motion_loader_uses_50hz_default_when_fps_missing(tmp_path: Path) -> None:
  motion_file = tmp_path / "motion_without_fps.npz"
  _write_motion(motion_file, include_fps=False)

  loader = MotionLoader(
    str(motion_file),
    torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  assert loader.fps == pytest.approx(50.0)


def test_multi_motion_loader_uses_50hz_default_when_fps_missing(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "motion_without_fps.npz"
  _write_motion(motion_file, include_fps=False)

  loader = MultiMotionLoader(
    [str(motion_file)],
    torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  assert loader.fps == pytest.approx(50.0)
  assert loader.fps_list[0] == pytest.approx(50.0)


def test_large_dataset_store_uses_50hz_default_when_fps_missing(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "motion_without_fps.npz"
  _write_motion(motion_file, include_fps=False)

  store = LargeDatasetMotionStore(
    [str(motion_file)],
    torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  assert store.fps == pytest.approx(50.0)
  assert store.fps_list[0] == pytest.approx(50.0)
  assert store.empty_fps_count == 1


def test_large_dataset_store_prints_metadata_progress(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  motion_a = tmp_path / "a.npz"
  motion_b = tmp_path / "b.npz"
  _write_motion(motion_a, include_fps=True)
  _write_motion(motion_b, include_fps=True)

  LargeDatasetMotionStore(
    [str(motion_a), str(motion_b)],
    torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  stdout = capsys.readouterr().out
  assert "metadata read start count=2" in stdout
  assert "metadata progress 2/2" in stdout


def test_large_dataset_store_can_read_metadata_in_parallel(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  motion_a = tmp_path / "a.npz"
  motion_b = tmp_path / "b.npz"
  _write_motion(motion_a, include_fps=False)
  _write_motion(motion_b, include_fps=True)

  store = LargeDatasetMotionStore(
    [str(motion_a), str(motion_b)],
    torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
    metadata_read_workers=2,
  )

  stdout = capsys.readouterr().out
  assert "metadata read start count=2 backend=parallel workers=2" in stdout
  assert "metadata progress 2/2" in stdout
  assert store.file_lengths.tolist() == [4, 4]
  assert store.fps_list[0] == pytest.approx(50.0)
  assert store.fps_list[1] == pytest.approx(60.0)
  assert store.empty_fps_count == 1


def test_multi_motion_loader_can_fk_legacy_30_body_motion_for_sp_asset(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "legacy_g1_motion.npz"
  _write_legacy_g1_motion(motion_file)

  loader = MultiMotionLoader(
    [str(motion_file)],
    SP_BODY_INDEXES,
    motion_type="mujoco",
    device="cpu",
    fk_from_joint_pos=True,
    fk_helper=_sp_fk_helper(),
  )

  assert loader.body_pos_w.shape == (6, len(SP_BODY_NAMES), 3)
  assert loader.body_quat_w.shape == (6, len(SP_BODY_NAMES), 4)
  assert torch.allclose(
    loader.body_quat_w.norm(dim=-1),
    torch.ones_like(loader.body_quat_w[..., 0]),
    atol=1.0e-5,
  )


def test_large_dataset_store_can_fk_legacy_30_body_motion_for_sp_asset(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "legacy_g1_motion.npz"
  _write_legacy_g1_motion(motion_file)

  store = LargeDatasetMotionStore(
    [str(motion_file)],
    SP_BODY_INDEXES,
    motion_type="mujoco",
    device="cpu",
    fk_from_joint_pos=True,
    fk_helper=_sp_fk_helper(),
  )

  loaded = store.load_motion_ids(torch.tensor([0], dtype=torch.long))

  assert loaded.body_pos_w.shape == (6, len(SP_BODY_NAMES), 3)
  assert loaded.body_quat_w.shape == (6, len(SP_BODY_NAMES), 4)
