from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.mdp.keypoints import SemanticKeypointResolver


def _quat_z(angle: float) -> torch.Tensor:
  return torch.tensor(
    [[math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]],
    dtype=torch.float32,
  )


def _asset() -> SimpleNamespace:
  data = SimpleNamespace(
    body_link_pos_w=torch.tensor([[[1.0, 2.0, 3.0]]]),
    body_link_quat_w=_quat_z(math.pi / 2.0).unsqueeze(1),
    body_link_lin_vel_w=torch.tensor([[[0.5, 1.0, 0.0]]]),
    body_link_ang_vel_w=torch.tensor([[[0.0, 0.0, 2.0]]]),
  )
  return SimpleNamespace(body_names=("parent",), data=data)


def test_semantic_keypoint_applies_rigid_offset_and_point_velocity() -> None:
  asset = _asset()
  resolver = SemanticKeypointResolver(
    asset,
    ("parent",),
    ({"name": "point", "body_name": "parent", "local_pos": (1.0, 0.0, 0.0)},),
  )

  point = resolver.current(asset)

  torch.testing.assert_close(point.pos_w, torch.tensor([[[1.0, 3.0, 3.0]]]))
  # omega x R*r = [0, 0, 2] x [0, 1, 0] = [-2, 0, 0]
  torch.testing.assert_close(
    point.lin_vel_w, torch.tensor([[[-1.5, 1.0, 0.0]]])
  )
  torch.testing.assert_close(point.ang_vel_w, asset.data.body_link_ang_vel_w)
  torch.testing.assert_close(point.quat_w, asset.data.body_link_quat_w)


def test_asset_and_reference_indices_are_resolved_independently() -> None:
  asset = _asset()
  resolver = SemanticKeypointResolver(
    asset,
    ("other", "reference_parent"),
    (
      {
        "name": "point",
        "asset_body_name": "parent",
        "reference_body_name": "reference_parent",
        "asset_local_pos": (0.0, 0.0, 0.0),
        "reference_local_pos": (0.25, 0.0, 0.0),
      },
    ),
  )
  pos = torch.zeros((1, 2, 2, 3))
  quat = torch.zeros((1, 2, 2, 4))
  quat[..., 0] = 1.0
  lin = torch.zeros_like(pos)
  ang = torch.zeros_like(pos)
  pos[:, :, 1] = torch.tensor([2.0, 0.0, 0.0])

  point = resolver.reference(pos, quat, lin, ang)

  assert point.pos_w.shape == (1, 2, 1, 3)
  torch.testing.assert_close(
    point.pos_w[..., 0, :],
    torch.tensor([[[2.25, 0.0, 0.0], [2.25, 0.0, 0.0]]]),
  )


def test_missing_parent_body_fails_at_resolver_construction() -> None:
  asset = _asset()
  try:
    SemanticKeypointResolver(
      asset,
      ("reference_parent",),
      ({"name": "point", "body_name": "missing"},),
    )
  except ValueError as error:
    assert "parent bodies are missing" in str(error)
  else:
    raise AssertionError("missing semantic keypoint parent must fail fast")
