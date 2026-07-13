from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .multi_commands import MotionCommand
from .rewards import _get_body_indexes, _relative_reference_body_poses

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  anchor_body_name: str | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if anchor_body_name is None:
    anchor_pos_w = command.anchor_pos_w
    robot_anchor_pos_w = command.robot_anchor_pos_w
  else:
    body_indexes = _get_body_indexes(command, (anchor_body_name,))
    if len(body_indexes) != 1:
      raise ValueError(
        f"Anchor body '{anchor_body_name}' is absent from the command reference."
      )
    body_index = body_indexes[0]
    anchor_pos_w = command.body_pos_w[:, body_index]
    robot_anchor_pos_w = command.robot_body_pos_w[:, body_index]
  return (
    torch.abs(anchor_pos_w[:, -1] - robot_anchor_pos_w[:, -1]) > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  threshold: float,
  anchor_body_name: str | None = None,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if anchor_body_name is None:
    anchor_quat_w = command.anchor_quat_w
    robot_anchor_quat_w = command.robot_anchor_quat_w
  else:
    body_indexes = _get_body_indexes(command, (anchor_body_name,))
    if len(body_indexes) != 1:
      raise ValueError(
        f"Anchor body '{anchor_body_name}' is absent from the command reference."
      )
    body_index = body_indexes[0]
    anchor_quat_w = command.body_quat_w[:, body_index]
    robot_anchor_quat_w = command.robot_body_quat_w[:, body_index]
  motion_projected_gravity_b = quat_apply_inverse(
    anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
  anchor_body_name: str | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  reference_pos_w, _ = _relative_reference_body_poses(
    command, body_indexes, anchor_body_name
  )
  error = torch.norm(
    reference_pos_w - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
  anchor_body_name: str | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  reference_pos_w, _ = _relative_reference_body_poses(
    command, body_indexes, anchor_body_name
  )
  error = torch.abs(
    reference_pos_w[..., -1] - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)
