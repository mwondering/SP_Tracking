"""Left-right symmetry transforms for the SP HEFT observation schema."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict

from sp_tracking.tasks.tracking.mdp.sp import (
  SP_FEET_BODY_NAMES,
  SP_KEYPOINT_BODY_NAMES,
  SP_TERMINATION_BODY_NAMES,
)


@dataclass
class _Transform:
  perm: torch.Tensor
  signs: torch.Tensor

  def __call__(self, value: torch.Tensor, *, sign: bool = True) -> torch.Tensor:
    result = value[..., self.perm.to(value.device)]
    return result * self.signs.to(value.device) if sign else result

  def repeat(self, count: int) -> "_Transform":
    return _cat([self] * count)


def _cat(transforms: list[_Transform]) -> _Transform:
  permutations = []
  signs = []
  offset = 0
  for transform in transforms:
    permutations.append(transform.perm + offset)
    signs.append(transform.signs)
    offset += transform.perm.numel()
  return _Transform(torch.cat(permutations), torch.cat(signs))


def _identity(size: int) -> _Transform:
  return _Transform(torch.arange(size), torch.ones(size))


def _component(signs) -> _Transform:
  signs = torch.as_tensor(signs, dtype=torch.float32)
  return _Transform(torch.arange(signs.numel()), signs)


def _named_transform(names, mapping, component_signs) -> _Transform:
  names = tuple(names)
  width = len(component_signs)
  perm = []
  signs = []
  for name in names:
    mapped = mapping[name]
    joint_sign = 1.0
    if isinstance(mapped, tuple):
      joint_sign, mapped = mapped
    source = names.index(mapped)
    perm.extend(source * width + index for index in range(width))
    signs.extend(float(joint_sign) * float(value) for value in component_signs)
  return _Transform(torch.as_tensor(perm), torch.as_tensor(signs))


def _event_output_transform(root, term_name: str, attributes: tuple[str, ...], mapping):
  """Build a transform in the exact column order exposed by a DR event."""
  event_manager = getattr(root, "event_manager", None)
  if event_manager is None:
    raise RuntimeError(
      f"SP symmetry requires the {term_name!r} event to mirror priv_critic"
    )
  func = event_manager.get_term_cfg(term_name).func
  transforms = []
  for attribute in attributes:
    names = tuple(getattr(func, attribute, ()))
    if names:
      transforms.append(_named_transform(names, mapping, (1.0,)))
  if not transforms:
    raise RuntimeError(
      f"SP symmetry could not resolve any {term_name!r} observation columns"
    )
  return _cat(transforms)


def _build(env):
  root = getattr(env, "unwrapped", env)
  asset = root.scene["robot"]
  joint_names = tuple(asset.cfg.joint_name_order)
  joint = _named_transform(
    joint_names, asset.cfg.joint_symmetry_mapping, (1.0,)
  )
  cart = _component((1.0, -1.0, 1.0))
  angular = _component((-1.0, 1.0, -1.0))
  rot6d = _component((1.0, -1.0, 1.0, -1.0, 1.0, -1.0))
  body_map = asset.cfg.spatial_symmetry_mapping
  key_cart = _named_transform(SP_KEYPOINT_BODY_NAMES, body_map, (1.0, -1.0, 1.0))
  key_rot = _named_transform(
    SP_KEYPOINT_BODY_NAMES, body_map, (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
  )
  termination_scalar = _named_transform(
    SP_TERMINATION_BODY_NAMES, body_map, (1.0,)
  )
  feet_cart = _named_transform(SP_FEET_BODY_NAMES, body_map, (1.0, -1.0, 1.0))
  feet_scalar = _named_transform(SP_FEET_BODY_NAMES, body_map, (1.0,))

  policy = _cat(
    [
      _identity(1),
      cart.repeat(12), rot6d.repeat(13),
      joint.repeat(13), joint.repeat(13),
      _identity(13), cart.repeat(13),
      angular.repeat(9), cart.repeat(9),
      joint.repeat(9), joint.repeat(9), joint.repeat(8),
    ]
  )
  priv = _cat(
    [
      joint.repeat(12), joint.repeat(12),
      _identity(12), cart.repeat(12),
      cart.repeat(12), rot6d.repeat(12), cart.repeat(12), angular.repeat(12),
      key_cart, key_rot, key_cart, _named_transform(
        SP_KEYPOINT_BODY_NAMES, body_map, (-1.0, 1.0, -1.0)
      ),
      key_cart.repeat(12), key_cart.repeat(12),
      key_rot.repeat(12), key_rot.repeat(12),
      cart.repeat(9), angular.repeat(9), cart.repeat(9),
      joint.repeat(9), joint.repeat(9),
      joint, joint,
      termination_scalar, termination_scalar, _identity(1),
      cart, cart, _identity(1),
      feet_cart, feet_scalar.repeat(3), feet_scalar,
    ]
  )
  joint_map = asset.cfg.joint_symmetry_mapping
  motor_dr = _event_output_transform(
    root,
    "motor_params_implicit",
    ("kp_names", "kd_names", "arm_names", "fric_names"),
    joint_map,
  )
  joint_offset_dr = _event_output_transform(
    root, "random_joint_offset", ("joint_names",), joint_map
  )
  priv_critic = _cat([motor_dr, _identity(3), joint_offset_dr, cart])
  action = joint
  return {"policy": policy, "priv": priv, "priv_critic": priv_critic}, action


def heft_teacher_symmetry(env, obs: TensorDict | None, actions: torch.Tensor | None):
  transforms, action_transform = _build(env)
  mirrored_obs = None
  if obs is not None:
    mirrored_obs = obs.clone()
    for key, transform in transforms.items():
      if key not in mirrored_obs.keys():
        continue
      value = mirrored_obs[key]
      if value.shape[-1] != transform.perm.numel():
        raise RuntimeError(
          f"SP symmetry dimension mismatch for {key}: "
          f"obs={value.shape[-1]} transform={transform.perm.numel()}"
        )
      mirrored_obs[key] = transform(value)
    mirrored_obs = torch.cat((obs, mirrored_obs), dim=0)
  mirrored_actions = None
  if actions is not None:
    mirrored = action_transform(actions)
    mirrored_actions = torch.cat((actions, mirrored), dim=0)
  return mirrored_obs, mirrored_actions
