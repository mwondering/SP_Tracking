from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from tensordict import TensorDict

from sp_tracking.tasks.tracking.mdp.keypoints import KeypointKinematics
from sp_tracking.tasks.tracking.mdp.spv4 import (
  RootFrameKeyBodyState,
  _key_body_error,
  _root_frame_state,
)
from sp_tracking.tasks.tracking.rl.spv4_models import SPV4KeyBodyActor


def _yaw(angle: float) -> torch.Tensor:
  return torch.tensor(
    [[math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)]],
    dtype=torch.float32,
  )


def _actor_observations(num_envs: int = 2) -> TensorDict:
  return TensorDict(
    {
      "actor_core": torch.randn(num_envs, 446),
      "estimator_history": torch.randn(num_envs, 6100),
      "estimator_target": torch.randn(num_envs, 4),
      "robot_key_body": torch.randn(num_envs, 195),
      "ref_key_body": torch.randn(num_envs, 195),
      "key_body_error": torch.randn(num_envs, 195),
    },
    batch_size=[num_envs],
  )


def _actor(obs: TensorDict) -> SPV4KeyBodyActor:
  return SPV4KeyBodyActor(
    obs,
    {
      "actor": [
        "actor_core",
        "estimator_history",
        "robot_key_body",
        "ref_key_body",
        "key_body_error",
      ]
    },
    "actor",
    3,
    hidden_dims=(32, 16),
    estimator_hidden_dims=(16, 8),
    obs_normalization=False,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def test_spv4_world_kinematics_are_converted_to_root_frame() -> None:
  root_pos = torch.tensor([[10.0, -4.0, 1.0]])
  root_quat = _yaw(math.pi / 2.0)
  root_lin_vel = torch.tensor([[4.0, 5.0, 6.0]])
  root_ang_vel = torch.tensor([[0.1, 0.2, 0.3]])
  local_pos = torch.tensor([[[1.0, 2.0, 3.0]]])
  local_quat = _yaw(-0.3).unsqueeze(1)
  local_lin_vel = torch.tensor([[[0.4, 0.5, 0.6]]])
  local_ang_vel = torch.tensor([[[-0.2, 0.3, 0.1]]])
  root_quat_key = root_quat.unsqueeze(1)
  keypoints = KeypointKinematics(
    pos_w=root_pos.unsqueeze(1) + quat_apply(root_quat_key, local_pos),
    quat_w=quat_mul(root_quat_key, local_quat),
    lin_vel_w=root_lin_vel.unsqueeze(1)
    + quat_apply(root_quat_key, local_lin_vel),
    ang_vel_w=root_ang_vel.unsqueeze(1)
    + quat_apply(root_quat_key, local_ang_vel),
  )

  actual = _root_frame_state(
    keypoints,
    root_pos,
    root_quat,
    root_lin_vel,
    root_ang_vel,
  )

  torch.testing.assert_close(actual.pos, local_pos)
  torch.testing.assert_close(actual.quat, local_quat)
  torch.testing.assert_close(actual.lin_vel, local_lin_vel)
  torch.testing.assert_close(actual.ang_vel, local_ang_vel)


def test_spv4_key_body_error_aligns_reference_root_and_is_zero_centered() -> None:
  count = 13
  robot_from_reference = _yaw(math.pi / 2.0)
  reference_from_robot = _yaw(-math.pi / 2.0).unsqueeze(1)
  robot_pos = torch.randn(1, count, 3)
  robot_lin_vel = torch.randn(1, count, 3)
  robot_ang_vel = torch.randn(1, count, 3)
  robot_quat = _yaw(0.4).unsqueeze(1).expand(-1, count, -1)
  reference_from_robot_quat = reference_from_robot.expand_as(robot_quat)
  reference = RootFrameKeyBodyState(
    pos=quat_apply(reference_from_robot, robot_pos),
    quat=quat_mul(reference_from_robot_quat, robot_quat),
    lin_vel=quat_apply(reference_from_robot, robot_lin_vel),
    ang_vel=quat_apply(reference_from_robot, robot_ang_vel),
  )
  robot = RootFrameKeyBodyState(
    pos=robot_pos,
    quat=robot_quat,
    lin_vel=robot_lin_vel,
    ang_vel=robot_ang_vel,
  )

  error = _key_body_error(robot, reference, robot_from_reference)

  assert error.shape == (1, 195)
  torch.testing.assert_close(error, torch.zeros_like(error), atol=1.0e-6, rtol=0.0)


def test_spv4_actor_preserves_spv3_estimator_gradient_boundary_and_export() -> None:
  obs = _actor_observations()
  actor = _actor(obs)

  assert actor.get_latent(obs).shape == (2, 1649)
  assert actor(obs).shape == (2, 3)
  exported = actor.as_onnx()
  flat_obs = torch.cat(
    [obs[name] for name in actor.obs_groups], dim=-1
  )
  assert flat_obs.shape == (2, 7131)
  assert exported.get_dummy_inputs()[0].shape == (1, 7131)
  torch.testing.assert_close(actor(obs), exported(flat_obs))

  actor.zero_grad()
  actor(obs).sum().backward()
  assert all(parameter.grad is None for parameter in actor.estimator.parameters())

  actor.zero_grad()
  height_mse, lin_vel_mse = actor.estimator_losses(obs)
  (height_mse + lin_vel_mse).backward()
  assert any(
    parameter.grad is not None for parameter in actor.estimator.parameters()
  )
