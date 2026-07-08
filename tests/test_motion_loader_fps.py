from pathlib import Path

import numpy as np
import pytest
import torch

from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  LargeDatasetMotionStore,
)
from sp_tracking.tasks.tracking.mdp.multi_commands import (
  MotionLoader,
  MultiMotionLoader,
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
