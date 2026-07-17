from __future__ import annotations

from types import SimpleNamespace

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from sp_tracking.tasks.tracking.mdp import spv3
from sp_tracking.tasks.tracking.rl.ppo import SPV3EstimatorPPO
from sp_tracking.tasks.tracking.rl.spv3_models import SPV3EstimatorActor


def _observations(num_envs: int = 3) -> TensorDict:
  return TensorDict(
    {
      "actor_core": torch.randn(num_envs, 446),
      "estimator_history": torch.randn(num_envs, 6100),
      "estimator_target": torch.randn(num_envs, 4),
      "policy": torch.randn(num_envs, 10),
      "priv": torch.randn(num_envs, 8),
    },
    batch_size=[num_envs],
  )


def _actor(obs: TensorDict, output_dim: int = 3) -> SPV3EstimatorActor:
  return SPV3EstimatorActor(
    obs,
    {
      "actor": ["actor_core", "estimator_history"],
      "critic": ["policy", "priv"],
    },
    "actor",
    output_dim,
    hidden_dims=(32, 16),
    estimator_hidden_dims=(16, 8),
    obs_normalization=False,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def test_spv3_actor_reuses_history_and_blocks_ppo_estimator_gradients() -> None:
  obs = _observations()
  actor = _actor(obs)

  assert actor(obs).shape == (3, 3)
  assert actor.get_latent(obs).shape == (3, 1064)
  exported = actor.as_onnx()
  assert exported.get_dummy_inputs()[0].shape == (1, 6546)
  flat_deploy_obs = torch.cat(
    (obs["actor_core"], obs["estimator_history"]), dim=-1
  )
  torch.testing.assert_close(actor(obs), exported(flat_deploy_obs))

  actor.zero_grad()
  actor(obs).sum().backward()
  assert all(parameter.grad is None for parameter in actor.estimator.parameters())

  actor.zero_grad()
  height_mse, lin_vel_mse = actor.estimator_losses(obs)
  (height_mse + lin_vel_mse).backward()
  assert any(
    parameter.grad is not None for parameter in actor.estimator.parameters()
  )


def test_spv3_root_velocity_error_uses_robot_frame() -> None:
  core = torch.zeros(1, 446)
  # robot_from_reference is a +90 degree yaw rotation.
  core[0, 12:18] = torch.tensor((0.0, 1.0, 0.0, -1.0, 0.0, 0.0))
  core[0, 47:50] = torch.tensor((1.0, 0.0, 0.0))

  actual = SPV3EstimatorActor._reference_lin_vel_in_robot_frame(core)

  torch.testing.assert_close(actual, torch.tensor([[0.0, 1.0, 0.0]]))


def test_spv3_estimator_ppo_reports_two_physical_mse_terms() -> None:
  obs = _observations(2)
  groups = {
    "actor": ["actor_core", "estimator_history"],
    "critic": ["policy", "priv"],
  }
  actor = _actor(obs)
  critic = MLPModel(
    obs,
    groups,
    "critic",
    1,
    hidden_dims=(16, 8),
    obs_normalization=False,
  )
  storage = RolloutStorage("rl", 2, 2, obs, [3], "cpu")
  algorithm = SPV3EstimatorPPO(
    actor,
    critic,
    storage,
    device="cpu",
    num_learning_epochs=1,
    num_mini_batches=1,
    learning_rate=1.0e-3,
    actor_learning_rate=1.0e-3,
    critic_learning_rate=1.0e-3,
    estimator_learning_rate=1.0e-4,
    schedule="fixed",
    desired_kl=None,
  )
  for _ in range(2):
    algorithm.act(obs)
    obs = _observations(2)
    algorithm.process_env_step(
      obs,
      torch.ones(2),
      torch.zeros(2),
      {},
    )
  algorithm.compute_returns(obs)

  losses = algorithm.update()

  assert losses["estimator_root_height_mse"] >= 0.0
  assert losses["estimator_root_lin_vel_mse"] >= 0.0
  assert losses["estimator_lr"] == 1.0e-4
  estimator_parameters = set(actor.estimator.parameters())
  assert estimator_parameters.intersection(algorithm.estimator_optimizer.state)
  assert not estimator_parameters.intersection(algorithm.optimizer.state)


def test_spv3_estimator_optimizer_checkpoint_is_optional_for_legacy_runs() -> None:
  obs = _observations(2)
  groups = {
    "actor": ["actor_core", "estimator_history"],
    "critic": ["policy", "priv"],
  }

  def make_algorithm(estimator_lr: float) -> SPV3EstimatorPPO:
    actor = _actor(obs)
    critic = MLPModel(
      obs,
      groups,
      "critic",
      1,
      hidden_dims=(16, 8),
      obs_normalization=False,
    )
    storage = RolloutStorage("rl", 2, 2, obs, [3], "cpu")
    return SPV3EstimatorPPO(
      actor,
      critic,
      storage,
      device="cpu",
      num_learning_epochs=1,
      num_mini_batches=1,
      actor_learning_rate=1.0e-3,
      critic_learning_rate=1.0e-3,
      estimator_learning_rate=estimator_lr,
      schedule="fixed",
      desired_kl=None,
    )

  source = make_algorithm(2.5e-5)
  source.estimator_optimizer.param_groups[0]["lr"] = 3.0e-5
  saved = source.save()

  resumed = make_algorithm(1.0e-4)
  resumed.load(saved, load_cfg=None, strict=True)
  assert resumed.estimator_learning_rate == 3.0e-5
  assert resumed.estimator_optimizer.param_groups[0]["lr"] == 3.0e-5

  legacy = dict(saved)
  legacy.pop(SPV3EstimatorPPO._ESTIMATOR_OPTIMIZER_STATE_KEY)
  resumed_legacy = make_algorithm(1.0e-4)
  resumed_legacy.load(legacy, load_cfg=None, strict=True)
  assert resumed_legacy.estimator_learning_rate == 1.0e-4
  assert resumed_legacy.estimator_optimizer.param_groups[0]["lr"] == 1.0e-4


def test_spv3_privileged_targets_use_environment_frame_and_body_velocity() -> None:
  root_pos = torch.tensor([[1.0, 2.0, 1.1], [4.0, 5.0, 0.9]])
  root_lin_vel_b = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.6]])
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=root_pos,
      root_link_lin_vel_b=root_lin_vel_b,
    )
  )

  class Scene:
    env_origins = torch.tensor([[0.0, 0.0, 0.2], [3.0, 4.0, 0.1]])

    def __getitem__(self, name: str):
      assert name == "robot"
      return robot

  env = SimpleNamespace(scene=Scene())

  torch.testing.assert_close(
    spv3.root_height_gt(env), torch.tensor([[0.9], [0.8]])
  )
  assert spv3.root_lin_vel_b_gt(env) is root_lin_vel_b
