"""SPV6 privileged physical parameters and recorded disturbance observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from mjlab.managers.observation_manager import ObservationTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


SPV6_GLOBAL_PHYSICS_DIM = 5
SPV6_SENSOR_PHYSICS_DIM = 29
SPV6_PHYSICS_DIM = SPV6_GLOBAL_PHYSICS_DIM + SPV6_SENSOR_PHYSICS_DIM
SPV6_PUSH_FRAME_DIM = 7
SPV6_PUSH_HISTORY_LENGTH = 50
SPV6_PUSH_HISTORY_DIM = SPV6_PUSH_FRAME_DIM * SPV6_PUSH_HISTORY_LENGTH


class physical_parameters:
  """Return calibrated or live BFM parameters in policy joint order."""

  def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
    self.mode: Literal["nominal", "actual"] = str(
      cfg.params.get("mode", "actual")
    )  # type: ignore[assignment]
    if self.mode not in {"nominal", "actual"}:
      raise ValueError(f"Unsupported SPV6 physical parameter mode: {self.mode}")
    self.env = env
    self.asset = env.scene["robot"]
    body_name = str(cfg.params.get("body_name", "torso_link"))
    geom_names = str(cfg.params.get("geom_names", r"^(left|right)_foot.*collision$"))
    body_ids, matched_bodies = self.asset.find_bodies(body_name)
    geom_ids, matched_geoms = self.asset.find_geoms(geom_names)
    if len(body_ids) != 1:
      raise ValueError(
        f"SPV6 physical parameters require one body for {body_name!r}, "
        f"got {matched_bodies}"
      )
    if not geom_ids:
      raise ValueError(
        f"SPV6 physical parameters matched no geoms for {geom_names!r}"
      )
    self.body_id = int(self.asset.indexing.body_ids[body_ids[0]].item())
    self.geom_ids = self.asset.indexing.geom_ids[
      torch.as_tensor(geom_ids, device=env.device, dtype=torch.long)
    ]
    action = env.action_manager.get_term("joint_pos")
    self.joint_ids = action.target_ids.to(device=env.device, dtype=torch.long)
    if self.joint_ids.numel() != SPV6_SENSOR_PHYSICS_DIM:
      raise ValueError(
        f"SPV6 expects {SPV6_SENSOR_PHYSICS_DIM} controlled joints, "
        f"got {self.joint_ids.numel()}"
      )
    self.nominal = self._read().detach().clone()

  def _read(self) -> torch.Tensor:
    model = self.env.sim.model
    com = model.body_ipos[:, self.body_id]
    mass = model.body_mass[:, self.body_id : self.body_id + 1]
    friction = model.geom_friction[:, self.geom_ids, 0].mean(
      dim=1, keepdim=True
    )
    encoder_bias = self.asset.data.encoder_bias.index_select(1, self.joint_ids)
    result = torch.cat((com, mass, friction, encoder_bias), dim=-1)
    if result.shape[-1] != SPV6_PHYSICS_DIM:
      raise RuntimeError(
        f"SPV6 physical vector has {result.shape[-1]} values, "
        f"expected {SPV6_PHYSICS_DIM}"
      )
    return result

  def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
    del env
    return self.nominal if self.mode == "nominal" else self._read()


def push_event_state(
  env: "ManagerBasedRlEnv", event_name: str = "push_robot"
) -> torch.Tensor:
  event_manager = getattr(env, "event_manager", None)
  if event_manager is None:
    return torch.zeros((env.num_envs, SPV6_PUSH_FRAME_DIM), device=env.device)
  try:
    event = event_manager.get_term_cfg(event_name).func
  except (KeyError, ValueError):
    return torch.zeros((env.num_envs, SPV6_PUSH_FRAME_DIM), device=env.device)
  observe = getattr(event, "observe", None)
  if not callable(observe):
    return torch.zeros((env.num_envs, SPV6_PUSH_FRAME_DIM), device=env.device)
  value = observe()
  if value.shape != (env.num_envs, SPV6_PUSH_FRAME_DIM):
    raise RuntimeError(
      f"SPV6 push event observation has shape {tuple(value.shape)}, "
      f"expected {(env.num_envs, SPV6_PUSH_FRAME_DIM)}"
    )
  return value
