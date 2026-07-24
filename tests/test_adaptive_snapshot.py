from __future__ import annotations

import json

import numpy as np
import torch

from sp_tracking.tasks.tracking.viewer.server import INDEX_HTML
from sp_tracking.tasks.tracking.viewer.snapshot import (
  PerRankAdaptiveBinSnapshotWriter,
)


def test_per_rank_snapshot_keeps_exact_motions_sorted_by_duration(tmp_path) -> None:
  snapshot_dir = tmp_path / "adaptive_bin_pool_view"
  motion_files = ("z.npz", "a.npz", "m.npz")
  file_lengths = torch.tensor([100, 50, 100])
  fps_list = (50.0, 25.0, 100.0)
  motion_bin_counts = torch.tensor([2, 1, 2])
  episode_count = torch.tensor(
    [
      [10.0, 11.0, 0.0],
      [20.0, 0.0, 0.0],
      [30.0, 31.0, 0.0],
    ]
  )
  failure_count = episode_count / 10.0

  writer = PerRankAdaptiveBinSnapshotWriter(
    snapshot_dir=snapshot_dir,
    motion_files=motion_files,
    file_lengths=file_lengths,
    fps_list=fps_list,
    motion_bin_counts=motion_bin_counts,
    bin_width_steps=50,
    failure_rate_window_iterations=1000,
    rank=0,
    world_size=2,
  )
  writer.write(
    episode_count=episode_count,
    failure_count=failure_count,
    iteration=25,
  )

  layout = json.loads((snapshot_dir / "layout.json").read_text())
  metadata = json.loads((snapshot_dir / "rank_000" / "latest.json").read_text())
  motions = json.loads((snapshot_dir / "rank_000" / "motions.json").read_text())
  access = np.fromfile(
    snapshot_dir / "rank_000" / "access.f32", dtype=np.float32
  ).reshape(3, 3)
  failure = np.fromfile(
    snapshot_dir / "rank_000" / "failure.f32", dtype=np.float32
  ).reshape(3, 3)

  assert layout["world_size"] == 2
  assert [item["rank"] for item in layout["ranks"]] == [0, 1]
  assert metadata["layout"] == "per_rank_exact_motion"
  assert metadata["motion_count"] == 3
  assert metadata["failure_rate_window_iterations"] == 1000
  assert [item["local_motion_id"] for item in motions["motions"]] == [2, 1, 0]
  assert [item["duration_seconds"] for item in motions["motions"]] == [
    1.0,
    2.0,
    2.0,
  ]
  assert [item["valid_bin_count"] for item in motions["motions"]] == [2, 1, 2]
  np.testing.assert_array_equal(access, episode_count[[2, 1, 0]].numpy())
  np.testing.assert_array_equal(failure, failure_count[[2, 1, 0]].numpy())


def test_per_rank_snapshot_writers_do_not_overwrite_other_ranks(tmp_path) -> None:
  snapshot_dir = tmp_path / "adaptive_bin_pool_view"
  common = {
    "snapshot_dir": snapshot_dir,
    "motion_files": ("motion.npz",),
    "file_lengths": (50,),
    "fps_list": (50.0,),
    "motion_bin_counts": (1,),
    "bin_width_steps": 50,
    "failure_rate_window_iterations": None,
    "world_size": 2,
  }
  rank_zero = PerRankAdaptiveBinSnapshotWriter(**common, rank=0)
  rank_one = PerRankAdaptiveBinSnapshotWriter(**common, rank=1)
  rank_zero.write(
    episode_count=torch.tensor([[1.0, 0.0]]),
    failure_count=torch.tensor([[0.0, 0.0]]),
    iteration=1,
  )
  rank_one.write(
    episode_count=torch.tensor([[2.0, 0.0]]),
    failure_count=torch.tensor([[1.0, 0.0]]),
    iteration=2,
  )

  rank_zero_access = np.fromfile(
    snapshot_dir / "rank_000" / "access.f32", dtype=np.float32
  )
  rank_one_access = np.fromfile(
    snapshot_dir / "rank_001" / "access.f32", dtype=np.float32
  )
  assert rank_zero_access.tolist() == [1.0, 0.0]
  assert rank_one_access.tolist() == [2.0, 0.0]


def test_viewer_supports_rank_layout_and_legacy_snapshot_protocols() -> None:
  assert "layout.json" in INDEX_HTML
  assert "per_rank_exact_motion" in INDEX_HTML
  assert "latest.json" in INDEX_HTML
  assert "bucket_start_motion_ids" in INDEX_HTML
