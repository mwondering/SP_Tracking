from __future__ import annotations

from hydra import compose, initialize_config_module
import torch
from mjlab.managers.event_manager import EventTermCfg
from tensordict import TensorDict

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.mdp.spv6 import (
  SPV6_PHYSICS_DIM,
  SPV6_PUSH_HISTORY_DIM,
)
from sp_tracking.tasks.tracking.mdp.randomizations import (
  recorded_push_by_setting_velocity,
)
from sp_tracking.tasks.tracking.rl.ppo import SPV6RmaPPO
from sp_tracking.tasks.tracking.rl.spv5_models import SPV5_POLICY_INPUT_DIM
from sp_tracking.tasks.tracking.rl.spv6_models import (
  SPV6_RMA_LATENT_DIM,
  SPV6RmaActor,
  SPV6RmaCritic,
)


KEYPOINT_SPECS = tuple(
  {"name": name, "body_name": body_name}
  for name, body_name in (
    ("left_hip", "left_hip_yaw_link"),
    ("left_knee", "left_knee_link"),
    ("left_foot", "left_ankle_roll_link"),
    ("right_hip", "right_hip_yaw_link"),
    ("right_knee", "right_knee_link"),
    ("right_foot", "right_ankle_roll_link"),
    ("head", "torso_link"),
    ("left_shoulder", "left_shoulder_yaw_link"),
    ("left_wrist", "left_wrist_roll_link"),
    ("left_hand", "left_wrist_yaw_link"),
    ("right_shoulder", "right_shoulder_yaw_link"),
    ("right_wrist", "right_wrist_roll_link"),
    ("right_hand", "right_wrist_yaw_link"),
  )
)


def _observations(num_envs: int = 2) -> TensorDict:
  reference_input = torch.randn(num_envs, 50, 38) * 0.05
  reference_input[..., 3:9] = torch.tensor(
    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
  )
  target = reference_input[:, -11:].clone()
  robot_key = torch.randn(num_envs, 195) * 0.05
  robot_key[:, 39:117] = torch.tensor(
    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
  ).repeat(num_envs, 13)
  push = torch.zeros(num_envs, 50, 7)
  push[:, 23, :6] = torch.randn(num_envs, 6) * 0.1
  push[:, 23, 6] = 1.0
  return TensorDict(
    {
      "robot_root_quat": torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(
        num_envs, 1
      ),
      # Existing SPV3/SPV5 history: 50 * (29+29+3+3+29+29), including torque.
      "estimator_history": torch.randn(num_envs, 6100),
      "estimator_target": torch.randn(num_envs, 4),
      "reference_encoder_input": reference_input.reshape(num_envs, -1),
      "reference_encoder_target": target.reshape(num_envs, -1),
      "robot_key_body": robot_key,
      "rma_physics_nominal": torch.randn(num_envs, SPV6_PHYSICS_DIM),
      "rma_physics_actual": torch.randn(num_envs, SPV6_PHYSICS_DIM),
      "rma_push_history": push.reshape(num_envs, -1),
      "policy": torch.randn(num_envs, 11),
      "priv": torch.randn(num_envs, 13),
    },
    batch_size=[num_envs],
  )


def _models(obs: TensorDict) -> tuple[SPV6RmaActor, SPV6RmaCritic]:
  actor = SPV6RmaActor(
    obs,
    {
      "actor": [
        "robot_root_quat",
        "estimator_history",
        "reference_encoder_input",
        "robot_key_body",
        "rma_physics_nominal",
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
  critic = SPV6RmaCritic(
    obs,
    {
      "critic": [
        "policy",
        "priv",
        "rma_physics_actual",
        "rma_push_history",
      ]
    },
    "critic",
    1,
    hidden_dims=(32, 16),
    obs_normalization=False,
  )
  return actor, critic


def test_spv6_actor_critic_latents_and_decoder_gradients() -> None:
  obs = _observations()
  actor, critic = _models(obs)

  actor_latents = actor.rma_latents(obs)
  critic_latents = critic.rma_latents(obs)
  assert [value.shape[-1] for value in actor_latents] == [8, 32, 16]
  assert [value.shape[-1] for value in critic_latents] == [8, 32, 16]
  assert all(value.abs().max() <= 1.0 for value in (*actor_latents, *critic_latents))
  assert actor.get_latent(obs).shape[-1] == SPV5_POLICY_INPUT_DIM + SPV6_RMA_LATENT_DIM
  assert actor(obs).shape == (2, 3)
  assert critic(obs).shape == (2, 1)

  actor.zero_grad()
  critic.zero_grad()
  alignment = sum(
    (actor_value - critic_value.detach()).square().mean()
    for actor_value, critic_value in zip(actor_latents, critic_latents, strict=True)
  )
  alignment.backward()
  assert any(parameter.grad is not None for parameter in actor.rma_global_head.parameters())
  assert all(parameter.grad is None for parameter in critic.global_encoder.parameters())

  actor.zero_grad()
  critic.zero_grad()
  physics_loss, push_loss, diagnostics = critic.reconstruction_losses(obs)
  (physics_loss + push_loss).backward()
  assert any(parameter.grad is not None for parameter in critic.global_encoder.parameters())
  assert any(parameter.grad is not None for parameter in critic.global_decoder.parameters())
  assert any(parameter.grad is not None for parameter in critic.push_decoder.parameters())
  assert "rma_push_mask_bce" in diagnostics


def test_recorded_push_exposes_only_the_current_step_delta() -> None:
  class _Data:
    root_link_vel_w = torch.zeros(2, 6)

  class _Asset:
    data = _Data()

    def write_root_link_velocity_to_sim(self, velocity, env_ids):
      self.data.root_link_vel_w[env_ids] = velocity

  class _Env:
    num_envs = 2
    device = "cpu"
    common_step_counter = 7
    scene = {"robot": _Asset()}

  ranges = {
    name: (float(index + 1), float(index + 1))
    for index, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw"))
  }
  cfg = EventTermCfg(
    func=recorded_push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 2.0),
    params={"velocity_range": ranges},
  )
  env = _Env()
  event = recorded_push_by_setting_velocity(cfg, env)
  event(env, torch.tensor([1]))

  observed = event.observe()
  torch.testing.assert_close(observed[0], torch.zeros(7))
  torch.testing.assert_close(
    observed[1], torch.tensor((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0))
  )
  env.common_step_counter += 1
  torch.testing.assert_close(event.observe(), torch.zeros(2, 7))


def test_spv6_ppo_combines_existing_and_rma_auxiliary_losses() -> None:
  obs = _observations()
  actor, critic = _models(obs)
  algorithm = object.__new__(SPV6RmaPPO)
  algorithm.actor = actor
  algorithm.critic = critic
  algorithm.estimator_root_height_loss_coef = 1.0
  algorithm.estimator_root_lin_vel_loss_coef = 1.0
  algorithm.reference_encoder_loss_coef = 1.0
  algorithm.rma_global_alignment_coef = 1.0
  algorithm.rma_sensor_alignment_coef = 0.5
  algorithm.rma_push_alignment_coef = 1.0
  algorithm.rma_physics_reconstruction_coef = 0.1
  algorithm.rma_push_reconstruction_coef = 0.1

  total, diagnostics = algorithm._auxiliary_loss(obs)

  assert total.ndim == 0
  assert {
    "estimator_root_height_mse",
    "reference_encoder_mse",
    "rma_alignment_global",
    "rma_alignment_sensor",
    "rma_alignment_push",
    "rma_reconstruction_global",
    "rma_reconstruction_sensor",
    "rma_push_mask_bce",
  }.issubset(diagnostics)


def test_spv6_task_uses_reset_dr_and_existing_torque_history() -> None:
  with initialize_config_module(
    version_base=None, config_module="sp_tracking.conf"
  ):
    cfg = compose(
      config_name="train",
      overrides=["task=tracking_bfm_spv6_actor_heft_critic_heft_reward"],
    )
  env = build_env_cfg(cfg.task)
  prepared = prepare_train_cfg(cfg)

  assert prepared.agent.actor.class_name.endswith(":SPV6RmaActor")
  assert prepared.agent.critic.class_name.endswith(":SPV6RmaCritic")
  assert prepared.agent.algorithm.class_name.endswith(":SPV6RmaPPO")
  assert prepared.agent.obs_groups["actor"][-1] == "rma_physics_nominal"
  assert prepared.agent.obs_groups["critic"][-2:] == (
    "rma_physics_actual",
    "rma_push_history",
  )
  assert env.observations["estimator_history"].terms["joint_torque"].history_length == 50
  assert env.observations["rma_push_history"].terms["push"].history_length == 50
  assert env.events["push_robot"].func.__name__ == (
    "recorded_push_by_setting_velocity"
  )
  for name in ("base_com", "base_mass", "encoder_bias", "foot_friction"):
    assert env.events[name].mode == "reset"
  assert env.events["foot_friction"].params["ranges"] == (0.2, 2.5)
  assert env.events["foot_friction"].params["shared_random"] is True
