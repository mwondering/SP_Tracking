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


def _asset_with_correction_parent() -> SimpleNamespace:
  identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
  data = SimpleNamespace(
    body_link_pos_w=torch.tensor([[[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]]),
    body_link_quat_w=torch.stack(
      (_quat_z(math.pi / 2.0)[0], identity[0])
    ).unsqueeze(0),
    body_link_lin_vel_w=torch.tensor([[[0.5, 1.0, 0.0], [8.0, 8.0, 8.0]]]),
    body_link_ang_vel_w=torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]]),
  )
  return SimpleNamespace(body_names=("parent", "correction_parent"), data=data)


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


def test_semantic_keypoint_applies_additive_correction_in_second_frame() -> None:
  asset = _asset_with_correction_parent()
  resolver = SemanticKeypointResolver(
    asset,
    ("parent", "correction_parent"),
    (
      {
        "name": "point",
        "body_name": "parent",
        "local_pos": (1.0, 0.0, 0.0),
        "correction_body_name": "correction_parent",
        "correction_local_pos": (0.5, 0.0, 0.0),
      },
    ),
  )

  current = resolver.current(asset)

  torch.testing.assert_close(current.pos_w, torch.tensor([[[1.5, 3.0, 3.0]]]))
  # Main: [0, 0, 2] x [0, 1, 0] = [-2, 0, 0].
  # Correction: [0, 0, 3] x [0.5, 0, 0] = [0, 1.5, 0].
  torch.testing.assert_close(
    current.lin_vel_w, torch.tensor([[[-1.5, 2.5, 0.0]]])
  )
  torch.testing.assert_close(
    current.quat_w, asset.data.body_link_quat_w[:, :1]
  )
  torch.testing.assert_close(
    current.ang_vel_w, asset.data.body_link_ang_vel_w[:, :1]
  )

  reference = resolver.reference(
    asset.data.body_link_pos_w.unsqueeze(1),
    asset.data.body_link_quat_w.unsqueeze(1),
    asset.data.body_link_lin_vel_w.unsqueeze(1),
    asset.data.body_link_ang_vel_w.unsqueeze(1),
  )
  torch.testing.assert_close(reference.pos_w[:, 0], current.pos_w)
  torch.testing.assert_close(reference.lin_vel_w[:, 0], current.lin_vel_w)


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


def test_missing_correction_parent_body_fails_at_resolver_construction() -> None:
  asset = _asset()
  try:
    SemanticKeypointResolver(
      asset,
      ("parent",),
      (
        {
          "name": "point",
          "body_name": "parent",
          "correction_body_name": "missing_correction_parent",
          "correction_local_pos": (0.005, 0.0, 0.0),
        },
      ),
    )
  except ValueError as error:
    assert "missing_correction_parent" in str(error)
  else:
    raise AssertionError("missing correction parent must fail fast")
