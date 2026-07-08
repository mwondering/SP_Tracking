from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class CompactBinPoolSnapshot:
  access_sum: np.ndarray
  failure_sum: np.ndarray
  valid_count: np.ndarray
  bucket_start_motion_ids: np.ndarray
  bucket_end_motion_ids: np.ndarray


def build_compact_bin_pool_snapshot(
  pool,
  *,
  num_buckets: int,
) -> CompactBinPoolSnapshot:
  """Compress a full adaptive bin pool shard into global motion-id buckets."""

  num_buckets = max(1, min(int(num_buckets), int(pool.num_files)))
  bin_count = int(pool.bin_count)
  device = pool.device

  with torch.no_grad():
    owned_motion_ids = pool.owned_motion_ids.to(device=device, dtype=torch.long)
    bucket_ids = torch.div(
      owned_motion_ids * num_buckets,
      int(pool.num_files),
      rounding_mode="floor",
    ).clamp_(0, num_buckets - 1)
    bin_ids = torch.arange(bin_count, dtype=torch.long, device=device)
    flat_indices = (bucket_ids.unsqueeze(1) * bin_count + bin_ids).reshape(-1)
    valid_mask = pool._bin_valid_mask_for(owned_motion_ids)

    access_values = torch.clamp_min(pool.bin_episode_count - float(pool.init_count), 0.0)
    failure_values = torch.clamp_min(pool.bin_failure_count - float(pool.init_count), 0.0)
    access_values = access_values.masked_fill(~valid_mask, 0.0).reshape(-1)
    failure_values = failure_values.masked_fill(~valid_mask, 0.0).reshape(-1)
    valid_values = valid_mask.to(dtype=torch.float32).reshape(-1)

    flat_size = num_buckets * bin_count
    access_sum = torch.zeros(flat_size, dtype=torch.float32, device=device)
    failure_sum = torch.zeros_like(access_sum)
    valid_count = torch.zeros_like(access_sum)
    access_sum.index_add_(0, flat_indices, access_values.to(dtype=torch.float32))
    failure_sum.index_add_(0, flat_indices, failure_values.to(dtype=torch.float32))
    valid_count.index_add_(0, flat_indices, valid_values)

    if dist.is_available() and dist.is_initialized():
      dist.all_reduce(access_sum, op=dist.ReduceOp.SUM)
      dist.all_reduce(failure_sum, op=dist.ReduceOp.SUM)
      dist.all_reduce(valid_count, op=dist.ReduceOp.SUM)

  bucket_ids_np = np.arange(num_buckets + 1, dtype=np.int64)
  bucket_edges = (bucket_ids_np * int(pool.num_files)) // num_buckets
  return CompactBinPoolSnapshot(
    access_sum=access_sum.reshape(num_buckets, bin_count).detach().cpu().numpy(),
    failure_sum=failure_sum.reshape(num_buckets, bin_count).detach().cpu().numpy(),
    valid_count=valid_count.reshape(num_buckets, bin_count)
    .to(dtype=torch.int32)
    .detach()
    .cpu()
    .numpy(),
    bucket_start_motion_ids=bucket_edges[:-1],
    bucket_end_motion_ids=bucket_edges[1:],
  )


class AdaptiveBinPoolSnapshotWriter:
  def __init__(
    self,
    *,
    snapshot_dir: str | os.PathLike[str],
    num_buckets: int,
    motion_files: list[str],
    manifest_file: str = "",
  ) -> None:
    self.snapshot_dir = Path(snapshot_dir)
    self.num_buckets = int(num_buckets)
    self.motion_files = list(motion_files)
    self.manifest_file = str(manifest_file)

  def write(self, pool, *, iteration: int) -> None:
    snapshot = build_compact_bin_pool_snapshot(pool, num_buckets=self.num_buckets)
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
      return
    self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    first_paths, last_paths = self._bucket_paths(
      snapshot.bucket_start_motion_ids, snapshot.bucket_end_motion_ids
    )

    self._write_array_atomic("access_sum.f32", snapshot.access_sum.astype(np.float32))
    self._write_array_atomic("failure_sum.f32", snapshot.failure_sum.astype(np.float32))
    self._write_array_atomic("valid_count.i32", snapshot.valid_count.astype(np.int32))
    metadata = {
      "version": 1,
      "iteration": int(iteration),
      "updated_at_unix": time.time(),
      "num_files": int(pool.num_files),
      "bucket_count": int(snapshot.access_sum.shape[0]),
      "bin_count": int(pool.bin_count),
      "bin_width_steps": int(pool.bin_width_steps),
      "init_count": float(pool.init_count),
      "manifest_file": self.manifest_file,
      "access_file": "access_sum.f32",
      "failure_file": "failure_sum.f32",
      "valid_file": "valid_count.i32",
      "bucket_start_motion_ids": snapshot.bucket_start_motion_ids.astype(int).tolist(),
      "bucket_end_motion_ids": snapshot.bucket_end_motion_ids.astype(int).tolist(),
      "bucket_first_paths": first_paths,
      "bucket_last_paths": last_paths,
    }
    self._write_json_atomic("latest.json", metadata)

  def _bucket_paths(
    self, bucket_starts: np.ndarray, bucket_ends: np.ndarray
  ) -> tuple[list[str], list[str]]:
    first_paths: list[str] = []
    last_paths: list[str] = []
    for start, end in zip(bucket_starts.tolist(), bucket_ends.tolist(), strict=True):
      if start >= end or start >= len(self.motion_files):
        first_paths.append("")
        last_paths.append("")
        continue
      last_index = min(int(end) - 1, len(self.motion_files) - 1)
      first_paths.append(self.motion_files[int(start)])
      last_paths.append(self.motion_files[last_index])
    return first_paths, last_paths

  def _write_array_atomic(self, filename: str, array: np.ndarray) -> None:
    path = self.snapshot_dir / filename
    tmp_path = path.with_name(f".{path.name}.tmp")
    array.tofile(tmp_path)
    os.replace(tmp_path, path)

  def _write_json_atomic(self, filename: str, data: dict) -> None:
    path = self.snapshot_dir / filename
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp_path, path)
