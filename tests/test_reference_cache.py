from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  LargeDatasetMotionSlotBuffer,
)
from sp_tracking.tasks.tracking.mdp.multi_commands import (
  REFERENCE_MOTION_FIELDS,
  MultiMotionCommand,
)


def _field_frames(frames: int, tail_shape: tuple[int, ...], offset: float) -> torch.Tensor:
  size = frames * max(torch.Size(tail_shape).numel(), 1)
  return torch.arange(size, dtype=torch.float32).reshape(frames, *tail_shape) + offset


def _fake_multimotion_command() -> MultiMotionCommand:
  command = object.__new__(MultiMotionCommand)
  command._env = SimpleNamespace(device="cpu", num_envs=3)
  command.cfg = SimpleNamespace(
    history_steps=0,
    future_steps=1,
    reference_cache_enabled=True,
    reference_cache_steps={
      field_name: (-2, 0, 1, 3) for field_name in REFERENCE_MOTION_FIELDS
    },
  )
  command.motion_idx = torch.tensor([0, 1, 0], dtype=torch.long)
  command.time_steps = torch.tensor([2, 1, 4], dtype=torch.long)
  command.motion = SimpleNamespace(
    file_lengths=torch.tensor([6, 5], dtype=torch.long),
    length_starts=torch.tensor([0, 6], dtype=torch.long),
  )
  total_frames = 11
  shapes = {
    "joint_pos": (2,),
    "joint_vel": (2,),
    "body_pos_w": (3, 3),
    "body_quat_w": (3, 4),
    "body_lin_vel_w": (3, 3),
    "body_ang_vel_w": (3, 3),
  }
  for index, (field_name, shape) in enumerate(shapes.items()):
    setattr(command.motion, field_name, _field_frames(total_frames, shape, index * 1000.0))
  command._initialize_reference_cache()
  return command


def test_multimotion_reference_cache_matches_uncached_gather() -> None:
  command = _fake_multimotion_command()

  for field_name in REFERENCE_MOTION_FIELDS:
    steps = (-2, 0, 3)
    cached = command.gather_reference(field_name, steps)
    absolute_steps = command.time_steps[:, None] + torch.tensor(steps)[None, :]
    direct = command._gather_motion_field(
      field_name, command.motion_idx, absolute_steps
    )
    torch.testing.assert_close(cached, direct, rtol=0.0, atol=0.0)

  first = command.gather_reference("body_pos_w", (-2, 0, 3))
  second = command.gather_reference("body_pos_w", (-2, 0, 3))
  assert first.data_ptr() == second.data_ptr()
  assert command._reference_cache_build_count == 1


def test_multimotion_reference_cache_invalidation_tracks_new_time_steps() -> None:
  command = _fake_multimotion_command()
  command.gather_reference("joint_pos", (0,))

  command.time_steps.add_(1)
  command._invalidate_reference_cache()
  cached = command.gather_reference("joint_pos", (0, 1))
  absolute_steps = command.time_steps[:, None] + torch.tensor((0, 1))[None, :]
  direct = command._gather_motion_field("joint_pos", command.motion_idx, absolute_steps)

  torch.testing.assert_close(cached, direct, rtol=0.0, atol=0.0)
  assert command._reference_cache_build_count == 2


def test_large_dataset_gather_many_matches_source_chunks() -> None:
  lengths = (3, 5, 4)
  shapes = {
    "joint_pos": (2,),
    "joint_vel": (2,),
    "body_pos_w": (3, 3),
    "body_quat_w": (3, 4),
    "body_lin_vel_w": (3, 3),
    "body_ang_vel_w": (3, 3),
  }
  chunks = {
    field_name: [
      _field_frames(length, shape, field_index * 1000.0 + slot * 100.0)
      for slot, length in enumerate(lengths)
    ]
    for field_index, (field_name, shape) in enumerate(shapes.items())
  }
  buffer = LargeDatasetMotionSlotBuffer(
    global_motion_ids=torch.arange(len(lengths)),
    chunks=chunks,
    file_lengths=torch.tensor(lengths),
    fps=50.0,
  )
  slot_ids = torch.tensor([2, 0, 1, 2], dtype=torch.long)
  time_steps = torch.tensor(
    [[0, 3], [2, 0], [4, 1], [2, 1]], dtype=torch.long
  )

  gathered = buffer.gather_many(REFERENCE_MOTION_FIELDS, slot_ids, time_steps)
  for field_name in REFERENCE_MOTION_FIELDS:
    expected = torch.stack(
      [
        torch.stack(
          [chunks[field_name][int(slot)][int(step)] for step in row]
        )
        for slot, row in zip(slot_ids, time_steps, strict=True)
      ]
    )
    torch.testing.assert_close(gathered[field_name], expected, rtol=0.0, atol=0.0)
