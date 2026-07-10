from pathlib import Path

import numpy as np
import pytest
import torch

from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  LargeDatasetMotionStore,
)


def _write_motion(path: Path) -> None:
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
    "fps": np.asarray([50.0], dtype=np.float32),
  }
  motion["body_quat_w"][..., 0] = 1.0
  np.savez(path, **motion)


def test_large_dataset_store_reports_nonfinite_motion_field(tmp_path: Path) -> None:
  motion_file = tmp_path / "bad_motion.npz"
  _write_motion(motion_file)
  with np.load(motion_file) as data:
    motion = {key: data[key] for key in data.files}
  motion["joint_pos"][1, 2] = np.nan
  np.savez(motion_file, **motion)

  store = LargeDatasetMotionStore(
    [str(motion_file)],
    torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  with pytest.raises(ValueError, match="Non-finite motion data.*joint_pos.*bad_motion"):
    store.load_motion_ids(torch.tensor([0], dtype=torch.long))
