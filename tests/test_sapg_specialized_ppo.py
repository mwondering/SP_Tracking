from copy import deepcopy

import torch
from tensordict import TensorDict

from sp_tracking.tasks.tracking.rl.ppo import (
  HeftTeacherPPO,
  SPV3EstimatorPPO,
)
from sp_tracking.tasks.tracking.rl.sapg.conditioning import (
  BlockGaussianDistribution,
  PolicyConditionedLinear,
)


class _Env:
  num_envs = 8
  num_actions = 3


def _mirror_batch(*, env, obs=None, actions=None):
  del env
  mirrored_obs = torch.cat((obs, obs), dim=0) if obs is not None else None
  mirrored_actions = (
    torch.cat((actions, -actions), dim=0) if actions is not None else None
  )
  return mirrored_obs, mirrored_actions


def _sapg_cfg() -> dict:
  return {
    "enabled": True,
    "compatibility": "official",
    "num_policy_blocks": 4,
    "local_parameter_dim": 8,
    "off_policy_ratio": 1,
    "exploration_type": "entropy",
    "entropy_coef_scale": 1.0,
    "value_eval_chunk_size": 8,
  }


def _common_config(actor: dict, critic: dict, algorithm: dict, groups: dict):
  return {
    "actor": actor,
    "critic": critic,
    "algorithm": {
      "value_loss_coef": 1.0,
      "use_clipped_value_loss": False,
      "clip_param": 0.2,
      "entropy_coef": 0.0,
      "num_learning_epochs": 1,
      "num_mini_batches": 2,
      "learning_rate": 1.0e-3,
      "schedule": "fixed",
      "gamma": 0.99,
      "lam": 0.95,
      "desired_kl": None,
      "max_grad_norm": 1.0,
      "sapg_cfg": _sapg_cfg(),
      **algorithm,
    },
    "obs_groups": groups,
    "num_steps_per_env": 2,
    "multi_gpu": None,
    "torch_compile_mode": None,
  }


def _heft_obs(value: float = 0.0) -> TensorDict:
  return TensorDict(
    {
      "policy": torch.full((8, 7), value),
      "priv": torch.full((8, 9), value),
      "priv_critic": torch.full((8, 4), value),
    },
    batch_size=[8],
  )


def _heft_config() -> dict:
  return _common_config(
    actor={
      "class_name": "sp_tracking.tasks.tracking.rl.heft_models:HeftTeacherActor",
      "hidden_dims": [16, 8],
      "activation": "mish",
      "obs_normalization": True,
      "privileged_latent_dim": 6,
      "init_std": [1.5, 1.2, 1.0],
      "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    },
    critic={
      "class_name": "sp_tracking.tasks.tracking.rl.heft_models:HeftTeacherCritic",
      "hidden_dims": [16, 8],
      "activation": "mish",
      "obs_normalization": True,
      "distribution_cfg": None,
    },
    algorithm={
      "class_name": "sp_tracking.tasks.tracking.rl.ppo:HeftTeacherPPO",
      "optimizer": "muon",
      "actor_learning_rate": 1.0e-3,
      "critic_learning_rate": 1.0e-3,
      "entropy_coef_start": 0.0,
      "entropy_coef_end": 0.0,
      "symmetry_cfg": {
        "data_augmentation_func": _mirror_batch,
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.1,
      },
    },
    groups={
      "actor": ["policy", "priv"],
      "critic": ["policy", "priv", "priv_critic"],
    },
  )


def test_heft_runs_sapg_update_with_muon_group_and_block_std() -> None:
  algorithm = HeftTeacherPPO.construct_algorithm(
    _heft_obs(), _Env(), deepcopy(_heft_config()), "cpu"
  )
  obs = _heft_obs()
  for step in range(2):
    algorithm.act(obs)
    next_obs = _heft_obs(float(step + 1))
    algorithm.process_env_step(
      next_obs, torch.ones(8), torch.zeros(8, dtype=torch.bool), {}
    )
    obs = next_obs
  algorithm.compute_returns(obs)
  losses = algorithm.update()

  assert isinstance(algorithm.actor.mlp[0], PolicyConditionedLinear)
  assert isinstance(algorithm.actor.distribution, BlockGaussianDistribution)
  assert algorithm.actor.distribution.std_param.shape == (4, 3)
  assert losses["actor_lr"] == 1.0e-3
  assert losses["critic_lr"] == 1.0e-3
  assert losses["symmetry"] >= 0.0


def _spv3_obs(value: float = 0.0) -> TensorDict:
  return TensorDict(
    {
      "actor_core": torch.full((8, 446), value),
      "estimator_history": torch.full((8, 6100), value),
      "estimator_target": torch.full((8, 4), value),
      "policy": torch.full((8, 7), value),
      "priv": torch.full((8, 9), value),
    },
    batch_size=[8],
  )


def _spv3_config() -> dict:
  return _common_config(
    actor={
      "class_name": "sp_tracking.tasks.tracking.rl.spv3_models:SPV3EstimatorActor",
      "hidden_dims": [16, 8],
      "activation": "elu",
      "obs_normalization": True,
      "estimator_hidden_dims": [16, 8],
      "estimator_activation": "elu",
      "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    },
    critic={
      "class_name": "sp_tracking.tasks.tracking.rl.heft_models:HeftTeacherCritic",
      "hidden_dims": [16, 8],
      "activation": "mish",
      "obs_normalization": True,
      "distribution_cfg": None,
    },
    algorithm={
      "class_name": "sp_tracking.tasks.tracking.rl.ppo:SPV3EstimatorPPO",
      "actor_learning_rate": 1.0e-3,
      "critic_learning_rate": 1.0e-3,
      "estimator_learning_rate": 1.0e-4,
    },
    groups={
      "actor": ["actor_core", "estimator_history"],
      "critic": ["policy", "priv"],
    },
  )


def test_spv3_runs_sapg_update_and_keeps_auxiliary_optimizer_separate() -> None:
  algorithm = SPV3EstimatorPPO.construct_algorithm(
    _spv3_obs(), _Env(), deepcopy(_spv3_config()), "cpu"
  )
  obs = _spv3_obs()
  for step in range(2):
    algorithm.act(obs)
    next_obs = _spv3_obs(float(step + 1) * 0.01)
    algorithm.process_env_step(
      next_obs, torch.ones(8), torch.zeros(8, dtype=torch.bool), {}
    )
    obs = next_obs
  algorithm.compute_returns(obs)
  losses = algorithm.update()

  assert losses["estimator_root_height_mse"] >= 0.0
  assert losses["estimator_root_lin_vel_mse"] >= 0.0
  assert losses["estimator_lr"] == 1.0e-4
