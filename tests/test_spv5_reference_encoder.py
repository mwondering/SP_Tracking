from __future__ import annotations

from hydra import compose, initialize_config_module
import torch
from tensordict import TensorDict

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.mdp.motion_fk import (
  finite_diff_torch,
  normalize,
  smooth_avg5_torch,
)
from sp_tracking.tasks.tracking.mdp.spv5 import (
  SPV5_REFERENCE_INPUT_DIM,
  SPV5_REFERENCE_INPUT_STEPS,
  SPV5_REFERENCE_SUPPORT_STEPS,
  SPV5_REFERENCE_TARGET_DIM,
)
from sp_tracking.tasks.tracking.rl.spv5_models import (
  SPV5_POLICY_CONTEXT_CACHE_DIM,
  SPV5_POLICY_CONTEXT_CACHE_GROUP,
  SPV5_POLICY_INPUT_DIM,
  SPV5_RAW_ACTOR_OBS_DIM,
  SPV5ReferenceEncoderActor,
  SPV5ReferenceKinematics,
  _support_angvel_from_quat,
)
from sp_tracking.tasks.tracking.rl.spv5_1_models import (
  SPV5_1_POLICY_CONTEXT_CACHE_DIM,
  SPV5_1_POLICY_CONTEXT_CACHE_GROUP,
  SPV5_1_POLICY_INPUT_DIM,
  SPV51ContactEstimatorActor,
)
from sp_tracking.tasks.tracking.rl.ppo import (
  SPV5ReferenceEncoderPPO,
  SPV51ContactEstimatorPPO,
)


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


def _spv5_1_observations(num_envs: int = 2) -> TensorDict:
  obs = _observations(num_envs)
  target = torch.tensor((1.0, 0.0)).repeat(num_envs, 1)
  obs.set("foot_contact_target", target)
  return obs


def _spv5_1_actor(obs: TensorDict) -> SPV51ContactEstimatorActor:
  return SPV51ContactEstimatorActor(
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
    estimator_hidden_dims=(16, 8, 4),
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


def test_spv5_seven_frame_key_body_fk_is_exact_at_current_frame() -> None:
  torch.manual_seed(7)
  batch = 3
  current = 3
  key_support = 7
  kinematics = SPV5ReferenceKinematics(KEYPOINT_SPECS, fps=50.0)
  helper = kinematics._fk_helper(torch.device("cpu"))
  joint_pos = torch.randn(batch, 11, 29) * 0.2

  full_pos, full_quat = helper.body_pose(joint_pos)
  short_pos, short_quat = helper.body_pose(joint_pos[:, :key_support])
  torch.testing.assert_close(short_pos, full_pos[:, :key_support])
  torch.testing.assert_close(short_quat, full_quat[:, :key_support])

  root_quat = normalize(torch.randn(batch, 11, 4))
  root_ang_vel = smooth_avg5_torch(
    _support_angvel_from_quat(root_quat, 50.0, dim=1), dim=1
  )
  full_lin_vel = smooth_avg5_torch(
    finite_diff_torch(full_pos, 50.0, dim=1)
    + torch.linalg.cross(
      root_ang_vel.unsqueeze(-2).expand_as(full_pos), full_pos, dim=-1
    ),
    dim=1,
  )
  short_lin_vel = smooth_avg5_torch(
    finite_diff_torch(short_pos, 50.0, dim=1)
    + torch.linalg.cross(
      root_ang_vel[:, :key_support]
      .unsqueeze(-2)
      .expand_as(short_pos),
      short_pos,
      dim=-1,
    ),
    dim=1,
  )
  full_ang_vel = smooth_avg5_torch(
    _support_angvel_from_quat(full_quat, 50.0, dim=1), dim=1
  )
  short_ang_vel = smooth_avg5_torch(
    _support_angvel_from_quat(short_quat, 50.0, dim=1), dim=1
  )

  torch.testing.assert_close(short_pos[:, current], full_pos[:, current])
  torch.testing.assert_close(short_quat[:, current], full_quat[:, current])
  torch.testing.assert_close(
    short_lin_vel[:, current], full_lin_vel[:, current]
  )
  torch.testing.assert_close(
    short_ang_vel[:, current], full_ang_vel[:, current]
  )


def test_spv5_policy_and_supervised_gradients_are_separated() -> None:
  obs = _observations()
  actor = _actor(obs)

  actor.zero_grad()
  actor(obs).sum().backward()
  assert all(parameter.grad is None for parameter in actor.estimator.parameters())
  assert all(
    parameter.grad is None for parameter in actor.reference_encoder.parameters()
  )


def test_spv5_rollout_context_cache_avoids_recomputing_reference() -> None:
  obs = _observations()
  actor = _actor(obs)
  uncached_latent = actor.get_latent(obs)
  context = actor.populate_policy_context_cache(obs)
  latent = actor.get_latent(obs)

  assert context.shape == (2, SPV5_POLICY_CONTEXT_CACHE_DIM)
  assert SPV5_POLICY_CONTEXT_CACHE_GROUP in obs
  torch.testing.assert_close(latent, uncached_latent)
  obs["reference_encoder_input"].normal_(mean=100.0, std=10.0)
  obs["robot_root_quat"].normal_()
  torch.testing.assert_close(actor.get_latent(obs), latent)

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
  assert prepared.agent.algorithm.estimator_learning_rate == 1.0e-4
  assert len(prepared.agent.actor.keypoint_specs) == 13
  assert prepared.agent.actor.reference_encoder_hidden_dims == (512, 256, 128)
  randomization = cfg.task.command.command.student_motion_randomization
  assert randomization.enable is True
  assert randomization.root_pos_noise_std == 0.005
  assert randomization.root_ori_noise_std == 0.02
  assert randomization.joint_pos_noise_std == 0.01
  runtime_randomization = env.commands["motion"].student_motion_randomization
  assert runtime_randomization["enable"] is True
  assert runtime_randomization["root_pos_noise_std"] == 0.005
  assert runtime_randomization["root_ori_noise_std"] == 0.02
  assert runtime_randomization["joint_pos_noise_std"] == 0.01
  assert "actor_core" not in env.observations
  assert "ref_key_body" not in env.observations
  assert "key_body_error" not in env.observations


def test_spv5_1_contact_estimator_cache_policy_and_export_contract() -> None:
  obs = _spv5_1_observations()
  actor = _spv5_1_actor(obs)

  root, contact_logits = actor.estimate_root_and_contact(obs)
  assert root.shape == (2, 4)
  assert contact_logits.shape == (2, 2)
  assert actor.get_latent(obs).shape == (2, SPV5_1_POLICY_INPUT_DIM)
  uncached_output = actor(obs)

  context = actor.populate_policy_context_cache(obs)
  assert context.shape == (2, SPV5_1_POLICY_CONTEXT_CACHE_DIM)
  assert SPV5_1_POLICY_CONTEXT_CACHE_GROUP in obs
  torch.testing.assert_close(actor(obs), uncached_output)

  flat = torch.cat([obs[name] for name in actor.obs_groups], dim=-1)
  exported = actor.as_onnx()
  assert exported.get_dummy_inputs()[0].shape == (1, SPV5_RAW_ACTOR_OBS_DIM)
  torch.testing.assert_close(exported(flat), uncached_output)


def test_spv5_1_contact_estimator_losses_train_both_heads_only_explicitly() -> None:
  obs = _spv5_1_observations()
  actor = _spv5_1_actor(obs)

  actor.zero_grad()
  actor(obs).sum().backward()
  assert all(parameter.grad is None for parameter in actor.estimator.parameters())

  actor.zero_grad()
  height_mse, lin_vel_mse, contact_bce, diagnostics = (
    actor.estimator_contact_losses(obs)
  )
  (height_mse + lin_vel_mse + contact_bce).backward()
  assert any(
    parameter.grad is not None
    for parameter in actor.estimator.shared_backbone.parameters()
  )
  assert any(
    parameter.grad is not None
    for parameter in actor.estimator.root_head.parameters()
  )
  assert any(
    parameter.grad is not None
    for parameter in actor.estimator.contact_head.parameters()
  )
  assert set(diagnostics) == {
    "estimator_foot_contact_accuracy",
    "estimator_foot_contact_precision",
    "estimator_foot_contact_recall",
    "estimator_foot_contact_f1",
    "estimator_foot_contact_target_rate",
    "estimator_foot_contact_pred_rate",
  }


def test_spv5_1_ppo_reports_contact_classification_terms() -> None:
  obs = _spv5_1_observations()
  actor = _spv5_1_actor(obs)
  algorithm = object.__new__(SPV51ContactEstimatorPPO)
  algorithm.actor = actor
  algorithm.estimator_root_height_loss_coef = 1.0
  algorithm.estimator_root_lin_vel_loss_coef = 1.0
  algorithm.estimator_foot_contact_loss_coef = 0.1
  algorithm.reference_encoder_loss_coef = 1.0

  total, diagnostics = algorithm._auxiliary_loss(obs)

  assert total.ndim == 0
  assert "estimator_foot_contact_bce" in diagnostics
  assert "estimator_foot_contact_accuracy" in diagnostics
  assert "estimator_root_height_mse" in diagnostics
  assert "reference_encoder_mse" in diagnostics


def test_spv5_1_task_adds_only_a_simulation_contact_target() -> None:
  with initialize_config_module(
    version_base=None, config_module="sp_tracking.conf"
  ):
    cfg = compose(
      config_name="train",
      overrides=["task=tracking_bfm_spv5_1_actor_heft_critic_heft_reward"],
    )
  prepared = prepare_train_cfg(cfg)
  env = build_env_cfg(cfg.task)

  assert "foot_contact_target" in env.observations
  assert "foot_contact_target" not in prepared.agent.obs_groups["actor"]
  assert "foot_contact_target" not in prepared.agent.obs_groups["critic"]
  assert prepared.agent.actor.class_name.endswith(
    ":SPV51ContactEstimatorActor"
  )
  assert prepared.agent.actor.foot_contact_target_group == (
    "foot_contact_target"
  )
  assert prepared.agent.algorithm.class_name.endswith(
    ":SPV51ContactEstimatorPPO"
  )
  assert prepared.agent.algorithm.estimator_foot_contact_loss_coef == 0.1
  assert env.events["base_com"].mode == "startup"
  assert env.events["base_mass"].mode == "startup"
  assert env.events["foot_friction"].mode == "startup"
