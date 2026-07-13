from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .multi_commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _reference_body_indices(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  if body_names is None:
    return list(range(len(command.cfg.body_names)))
  configured_names = tuple(command.cfg.body_names)
  missing = [name for name in body_names if name not in configured_names]
  if missing:
    raise ValueError(
      "Requested observation body names are absent from the command reference: "
      f"{missing}"
    )
  return [configured_names.index(name) for name in body_names]


def _anchor_pose(
  command: MotionCommand, anchor_body_name: str | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return reference and robot anchor poses for an optional named view."""
  if anchor_body_name is None:
    return (
      command.anchor_pos_w,
      command.anchor_quat_w,
      command.robot_anchor_pos_w,
      command.robot_anchor_quat_w,
    )
  body_index = _reference_body_indices(command, (anchor_body_name,))[0]
  return (
    command.body_pos_w[:, body_index],
    command.body_quat_w[:, body_index],
    command.robot_body_pos_w[:, body_index],
    command.robot_body_quat_w[:, body_index],
  )


def motion_anchor_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  anchor_body_name: str | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ref_pos_w, ref_quat_w, robot_pos_w, robot_quat_w = _anchor_pose(
    command, anchor_body_name
  )

  pos, _ = subtract_frame_transforms(
    robot_pos_w,
    robot_quat_w,
    ref_pos_w,
    ref_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  anchor_body_name: str | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ref_pos_w, ref_quat_w, robot_pos_w, robot_quat_w = _anchor_pose(
    command, anchor_body_name
  )

  _, ori = subtract_frame_transforms(
    robot_pos_w,
    robot_quat_w,
    ref_pos_w,
    ref_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
  anchor_body_name: str | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indices = _reference_body_indices(command, body_names)
  _, _, robot_anchor_pos_w, robot_anchor_quat_w = _anchor_pose(
    command, anchor_body_name
  )

  num_bodies = len(body_indices)
  pos_b, _ = subtract_frame_transforms(
    robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w[:, body_indices],
    command.robot_body_quat_w[:, body_indices],
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
  anchor_body_name: str | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indices = _reference_body_indices(command, body_names)
  _, _, robot_anchor_pos_w, robot_anchor_quat_w = _anchor_pose(
    command, anchor_body_name
  )

  num_bodies = len(body_indices)
  _, ori_b = subtract_frame_transforms(
    robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w[:, body_indices],
    command.robot_body_quat_w[:, body_indices],
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
