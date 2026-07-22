import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  LargeDatasetMotionStore,
)
from sp_tracking.tasks.tracking.mdp.motion_fk import (
  MotionFKHelper,
  angvel_from_quat_wxyz_torch,
  finite_diff_torch,
  smooth_avg5_torch,
)
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


def _write_legacy_g1_motion(
  path: Path,
  *,
  joint_pos: np.ndarray | None = None,
  joint_vel: np.ndarray | None = None,
) -> None:
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
  if joint_pos is not None:
    assert joint_pos.shape == motion["joint_pos"].shape
    motion["joint_pos"] = joint_pos.astype(np.float32)
  if joint_vel is not None:
    assert joint_vel.shape == motion["joint_vel"].shape
    motion["joint_vel"] = joint_vel.astype(np.float32)
  motion["body_pos_w"][:, 0, 2] = 0.75
  motion["body_quat_w"][..., 0] = 1.0
  np.savez(path, **motion)


def _sp_fk_helper() -> MotionFKHelper:
  return MotionFKHelper.from_mjcf_path(
    xml_path="src/sp_tracking/assets/robots/g1_sp_tracking/g1.xml",
    dataset_joint_names=_MUJOCO_JOINT_NAMES,
    output_body_names=SP_BODY_NAMES,
    device="cpu",
  )


def test_motion_fk_analytic_kinematics_matches_finite_difference() -> None:
  torch.manual_seed(11)
  helper = _sp_fk_helper()
  joint_pos = torch.randn(3, len(_MUJOCO_JOINT_NAMES)) * 0.15
  joint_vel = torch.randn_like(joint_pos) * 0.5
  root_ang_vel = torch.randn(3, 3) * 0.3

  pos, quat, lin_vel, ang_vel = helper.body_kinematics(
    joint_pos,
    joint_vel,
    root_ang_vel,
  )
  dt = 2.0e-4
  sample_pos, sample_quat = helper.body_pose(
    torch.stack(
      (joint_pos - dt * joint_vel, joint_pos, joint_pos + dt * joint_vel)
    )
  )
  expected_lin_vel = (sample_pos[2] - sample_pos[0]) / (2.0 * dt)
  expected_lin_vel += torch.linalg.cross(
    root_ang_vel.unsqueeze(1).expand_as(pos),
    pos,
    dim=-1,
  )
  relative_ang_vel = angvel_from_quat_wxyz_torch(
    sample_quat,
    fps=1.0 / dt,
    dim=0,
  )[1]
  expected_ang_vel = relative_ang_vel + root_ang_vel.unsqueeze(1)

  torch.testing.assert_close(quat, sample_quat[1])
  torch.testing.assert_close(
    lin_vel,
    expected_lin_vel,
    atol=8.0e-4,
    rtol=4.0e-3,
  )
  torch.testing.assert_close(
    ang_vel,
    expected_ang_vel,
    atol=8.0e-4,
    rtol=4.0e-3,
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


def test_large_dataset_store_can_read_metadata_with_process_backend(
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
    metadata_read_backend="process",
    metadata_read_chunksize=1,
  )

  stdout = capsys.readouterr().out
  assert "metadata read start count=2 backend=process workers=2 chunksize=1" in stdout
  assert "metadata progress 2/2" in stdout
  assert store.file_lengths.tolist() == [4, 4]
  assert store.fps_list[0] == pytest.approx(50.0)
  assert store.fps_list[1] == pytest.approx(60.0)
  assert store.empty_fps_count == 1


def test_large_dataset_store_writes_and_reuses_json_metadata_cache(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  motion_a = tmp_path / "a.npz"
  motion_b = tmp_path / "b.npz"
  _write_motion(motion_a, include_fps=True)
  _write_motion(motion_b, include_fps=False)
  cache_file = tmp_path / "metadata.json"

  first_store = LargeDatasetMotionStore(
    [str(motion_a), str(motion_b)],
    torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
    metadata_cache_file=str(cache_file),
  )

  assert first_store.file_lengths.tolist() == [4, 4]
  with cache_file.open(encoding="utf-8") as f:
    payload = json.load(f)
  assert payload["version"] == 1
  assert payload["num_files"] == 2
  assert payload["file_lengths"] == [4, 4]
  assert payload["fps_values"] == pytest.approx([60.0, 50.0])
  assert payload["empty_fps_count"] == 1

  def fail_metadata_read(self):
    raise AssertionError("metadata cache should be reused")

  monkeypatch.setattr(
    LargeDatasetMotionStore,
    "_read_motion_metadata_from_files",
    fail_metadata_read,
  )

  second_store = LargeDatasetMotionStore(
    [str(motion_a), str(motion_b)],
    torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
    metadata_cache_file=str(cache_file),
  )

  assert second_store.file_lengths.tolist() == [4, 4]
  assert second_store.fps_list[0] == pytest.approx(60.0)
  assert second_store.fps_list[1] == pytest.approx(50.0)


def test_large_dataset_store_prints_motion_chunk_load_progress(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  motion_files = []
  for index in range(100):
    motion_file = tmp_path / f"motion_{index}.npz"
    _write_motion(motion_file, include_fps=True)
    motion_files.append(str(motion_file))

  store = LargeDatasetMotionStore(
    motion_files,
    torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )
  capsys.readouterr()

  store.load_motion_chunks(torch.arange(len(motion_files), dtype=torch.long))

  stdout = capsys.readouterr().out
  assert "load_motion_chunks start count=100" in stdout
  assert "load_motion_chunks progress 100/100" in stdout
  assert "file=" in stdout
  assert "load_motion_chunks done count=100" in stdout


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


@pytest.mark.parametrize("loader_kind", ("single", "multi", "large"))
def test_sp_joint_velocity_reconstruction_matches_reference(
  tmp_path: Path,
  loader_kind: str,
) -> None:
  frames = 6
  joint_pos = np.zeros((frames, 29), dtype=np.float32)
  joint_pos[:, 0] = np.asarray((0.0, 0.1, 0.4, 0.9, 1.6, 2.5), dtype=np.float32)
  raw_joint_vel = np.full((frames, 29), 123.0, dtype=np.float32)
  motion_file = tmp_path / "legacy_g1_motion.npz"
  _write_legacy_g1_motion(
    motion_file,
    joint_pos=joint_pos,
    joint_vel=raw_joint_vel,
  )
  expected = smooth_avg5_torch(
    finite_diff_torch(torch.from_numpy(joint_pos), 50.0, dim=0), dim=0
  )
  kwargs = {
    "motion_type": "mujoco",
    "device": "cpu",
    "fk_from_joint_pos": True,
    "recompute_joint_vel_from_joint_pos": True,
    "fk_helper": _sp_fk_helper(),
  }

  if loader_kind == "single":
    joint_vel = MotionLoader(str(motion_file), SP_BODY_INDEXES, **kwargs).joint_vel
  elif loader_kind == "multi":
    joint_vel = MultiMotionLoader([str(motion_file)], SP_BODY_INDEXES, **kwargs).joint_vel
  else:
    store = LargeDatasetMotionStore([str(motion_file)], SP_BODY_INDEXES, **kwargs)
    joint_vel = store.load_motion_ids(torch.tensor([0], dtype=torch.long)).joint_vel

  torch.testing.assert_close(joint_vel, expected, rtol=0.0, atol=0.0)
  assert not torch.equal(joint_vel, torch.from_numpy(raw_joint_vel))


def test_joint_velocity_reconstruction_is_opt_in(tmp_path: Path) -> None:
  joint_pos = np.zeros((6, 29), dtype=np.float32)
  raw_joint_vel = np.full((6, 29), 123.0, dtype=np.float32)
  motion_file = tmp_path / "legacy_g1_motion.npz"
  _write_legacy_g1_motion(
    motion_file,
    joint_pos=joint_pos,
    joint_vel=raw_joint_vel,
  )

  loader = MotionLoader(
    str(motion_file),
    torch.tensor([0], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  torch.testing.assert_close(loader.joint_vel, torch.from_numpy(raw_joint_vel))
