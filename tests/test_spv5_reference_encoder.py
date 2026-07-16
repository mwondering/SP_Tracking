from __future__ import annotations

from hydra import compose, initialize_config_module
import torch
from tensordict import TensorDict

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.mdp.spv5 import (
  SPV5_REFERENCE_INPUT_DIM,
  SPV5_REFERENCE_INPUT_STEPS,
  SPV5_REFERENCE_SUPPORT_STEPS,
  SPV5_REFERENCE_TARGET_DIM,
)
from sp_tracking.tasks.tracking.rl.spv5_models import (
  SPV5_POLICY_INPUT_DIM,
  SPV5_RAW_ACTOR_OBS_DIM,
  SPV5ReferenceEncoderActor,
)
from sp_tracking.tasks.tracking.rl.ppo import SPV5ReferenceEncoderPPO


KEYPOINT_SPECS = (
  {"name": "left_hip", "body_name": "left_hip_yaw_link"},
  {"name": "left_knee", "body_name": "left_knee_link"},
  {"name": "left_foot", "body_name": "left_ankle_roll_link"},
  {"name": "right_hip", "body_name": "right_hip_yaw_link"},
  {"name": "right_knee", "body_name": "right_knee_link"},
  {"name": "right_foot", "body_name": "right_ankle_roll_link"},
  {
    "name": "head",
    "body_name": "torso_link",
    "local_pos": (0.01, 0.0, 0.41),
  },
  {"name": "left_shoulder", "body_name": "left_shoulder_yaw_link"},
  {"name": "left_wrist", "body_name": "left_wrist_roll_link"},
  {
    "name": "left_hand",
    "body_name": "left_wrist_yaw_link",
    "local_pos": (0.116, 0.0, 0.0),
    "correction_body_name": "left_wrist_pitch_link",
    "correction_local_pos": (0.005, 0.0, 0.0),
  },
  {"name": "right_shoulder", "body_name": "right_shoulder_yaw_link"},
  {"name": "right_wrist", "body_name": "right_wrist_roll_link"},
  {
    "name": "right_hand",
    "body_name": "right_wrist_yaw_link",
    "local_pos": (0.116, 0.0, 0.0),
    "correction_body_name": "right_wrist_pitch_link",
    "correction_local_pos": (0.005, 0.0, 0.0),
  },
)


def _observations(num_envs: int = 2) -> TensorDict:
  reference_input = torch.randn(num_envs, 50, 38) * 0.05
  reference_input[..., 3:9] = torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
  reference_input[:, :, 0] += torch.arange(50).float() * 0.01
  target = reference_input[:, -11:].clone()
  target[..., :3] += torch.randn_like(target[..., :3]) * 0.02
  target[..., 9:] += torch.randn_like(target[..., 9:]) * 0.03

  robot_key = torch.randn(num_envs, 195) * 0.05
  robot_key[:, 39:117] = torch.tensor(
    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
  ).repeat(num_envs, 13)
  return TensorDict(
    {
      "robot_root_quat": torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(
        num_envs, 1
      ),
      "estimator_history": torch.randn(num_envs, 6100),
      "estimator_target": torch.randn(num_envs, 4),
      "reference_encoder_input": reference_input.reshape(num_envs, -1),
      "reference_encoder_target": target.reshape(num_envs, -1),
      "robot_key_body": robot_key,
    },
    batch_size=[num_envs],
  )


def _actor(obs: TensorDict) -> SPV5ReferenceEncoderActor:
  return SPV5ReferenceEncoderActor(
    obs,
    {
      "actor": [
        "robot_root_quat",
        "estimator_history",
        "reference_encoder_input",
        "robot_key_body",
      ]
    },
    "actor",
    3,
    hidden_dims=(32, 16),
    estimator_hidden_dims=(16, 8),
    reference_encoder_hidden_dims=(32, 16),
    obs_normalization=False,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    keypoint_specs=KEYPOINT_SPECS,
  )


def test_spv5_window_contract_and_zero_initialized_residual() -> None:
  assert SPV5_REFERENCE_INPUT_STEPS == tuple(range(-42, 8))
  assert SPV5_REFERENCE_SUPPORT_STEPS == tuple(range(-3, 8))
  assert SPV5_REFERENCE_INPUT_DIM == 1900
  assert SPV5_REFERENCE_TARGET_DIM == 418

  obs = _observations()
  actor = _actor(obs)
  torch.testing.assert_close(
    actor.encode_reference(obs),
    obs["reference_encoder_input"][:, -SPV5_REFERENCE_TARGET_DIM:],
  )
  assert actor.get_latent(obs).shape == (2, SPV5_POLICY_INPUT_DIM)
  assert actor(obs).shape == (2, 3)
  assert actor.obs_dim == SPV5_RAW_ACTOR_OBS_DIM == 8199


def test_spv5_policy_and_supervised_gradients_are_separated() -> None:
  obs = _observations()
  actor = _actor(obs)

  actor.zero_grad()
  actor(obs).sum().backward()
  assert all(parameter.grad is None for parameter in actor.estimator.parameters())
  assert all(
    parameter.grad is None for parameter in actor.reference_encoder.parameters()
  )

  actor.zero_grad()
  reference_loss, diagnostics = actor.reference_encoder_losses(obs)
  reference_loss.backward()
  assert any(
    parameter.grad is not None for parameter in actor.reference_encoder.parameters()
  )
  assert all(parameter.grad is None for parameter in actor.estimator.parameters())
  assert set(diagnostics) == {
    "reference_encoder_mse",
    "reference_root_pos_mse",
    "reference_root_rot6d_mse",
    "reference_joint_pos_mse",
  }

  actor.zero_grad()
  height_mse, lin_vel_mse = actor.estimator_losses(obs)
  (height_mse + lin_vel_mse).backward()
  assert any(parameter.grad is not None for parameter in actor.estimator.parameters())
  assert all(
    parameter.grad is None for parameter in actor.reference_encoder.parameters()
  )


def test_spv5_flat_export_reproduces_actor_output() -> None:
  obs = _observations()
  actor = _actor(obs)
  flat = torch.cat([obs[name] for name in actor.obs_groups], dim=-1)
  exported = actor.as_onnx()

  assert exported.get_dummy_inputs()[0].shape == (1, 8199)
  torch.testing.assert_close(actor(obs), exported(flat))


def test_spv5_ppo_reports_estimator_and_reference_mse_terms() -> None:
  obs = _observations()
  actor = _actor(obs)
  algorithm = object.__new__(SPV5ReferenceEncoderPPO)
  algorithm.actor = actor
  algorithm.estimator_root_height_loss_coef = 1.0
  algorithm.estimator_root_lin_vel_loss_coef = 1.0
  algorithm.reference_encoder_loss_coef = 1.0

  total, diagnostics = algorithm._auxiliary_loss(obs)

  assert total.ndim == 0
  assert set(diagnostics) == {
    "estimator_root_height_mse",
    "estimator_root_lin_vel_mse",
    "reference_encoder_mse",
    "reference_root_pos_mse",
    "reference_root_rot6d_mse",
    "reference_joint_pos_mse",
  }


def test_spv5_task_exposes_only_encoded_reference_to_actor() -> None:
  with initialize_config_module(
    version_base=None, config_module="sp_tracking.conf"
  ):
    cfg = compose(
      config_name="train",
      overrides=["task=tracking_bfm_spv5_actor_heft_critic_heft_reward"],
    )
  prepared = prepare_train_cfg(cfg)
  env = build_env_cfg(cfg.task)

  assert tuple(env.observations) == (
    "policy",
    "priv",
    "estimator_history",
    "estimator_target",
    "robot_key_body",
    "robot_root_quat",
    "reference_encoder_input",
    "reference_encoder_target",
  )
  assert prepared.agent.obs_groups == {
    "actor": (
      "robot_root_quat",
      "estimator_history",
      "reference_encoder_input",
      "robot_key_body",
    ),
    "critic": ("policy", "priv"),
  }
  assert prepared.agent.actor.class_name.endswith(
    ":SPV5ReferenceEncoderActor"
  )
  assert prepared.agent.algorithm.class_name.endswith(
    ":SPV5ReferenceEncoderPPO"
  )
  assert len(prepared.agent.actor.keypoint_specs) == 13
  assert "actor_core" not in env.observations
  assert "ref_key_body" not in env.observations
  assert "key_body_error" not in env.observations
