from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import sp as sp_mdp
from .spv1 import _root_reference

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ref_root_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  root_body_name: str | None = None,
) -> torch.Tensor:
  """Reference pelvis height at the current and next six control steps."""
  pos_w = _root_reference(env, command_name, "body_pos_w", root_body_name)
  return pos_w[..., 2].reshape(env.num_envs, -1)


def ref_root_lin_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  root_body_name: str | None = None,
) -> torch.Tensor:
  """Reference linear velocity in each reference pelvis's own local frame."""
  quat_w = _root_reference(env, command_name, "body_quat_w", root_body_name)
  lin_vel_w = _root_reference(
    env, command_name, "body_lin_vel_w", root_body_name
  )
  return sp_mdp._quat_apply_inverse(quat_w, lin_vel_w).reshape(env.num_envs, -1)
