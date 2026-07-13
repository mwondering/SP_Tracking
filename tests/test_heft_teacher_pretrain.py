from __future__ import annotations

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from sp_tracking.tasks.tracking.rl.heft_models import (
  HeftTeacherActor,
  HeftTeacherCritic,
)
from sp_tracking.tasks.tracking.rl.ppo import HeftTeacherPPO


def _observations(num_envs: int = 4) -> TensorDict:
  return TensorDict(
    {
      "policy": torch.randn(num_envs, 17),
      "priv": torch.randn(num_envs, 23),
      "priv_critic": torch.randn(num_envs, 5),
    },
    batch_size=[num_envs],
  )


def test_heft_teacher_models_use_privileged_encoder_and_vector_std() -> None:
  obs = _observations()
  groups = {
    "actor": ["policy", "priv"],
    "critic": ["policy", "priv", "priv_critic"],
  }
  init_std = [1.5, 1.2, 1.0]
  actor = HeftTeacherActor(obs, groups, "actor", 3, init_std=init_std)
  critic = HeftTeacherCritic(obs, groups, "critic", 1)

  assert actor.encoder_priv[-1].out_features == 256
  assert actor(obs).shape == (4, 3)
  assert critic(obs).shape == (4, 1)
  assert actor.as_onnx().input_names == ["policy_priv"]
  torch.testing.assert_close(actor.distribution.std_param, torch.tensor(init_std))

  with torch.no_grad():
    actor.distribution.std_param.fill_(2.0)
  actor.clamp_std()
  torch.testing.assert_close(actor.distribution.std_param, torch.tensor(init_std))


def test_heft_teacher_ppo_constructs_muon_group_and_schedules() -> None:
  obs = _observations(2)
  groups = {
    "actor": ["policy", "priv"],
    "critic": ["policy", "priv", "priv_critic"],
  }
  actor = HeftTeacherActor(obs, groups, "actor", 3, init_std=[1.5, 1.2, 1.0])
  critic = HeftTeacherCritic(obs, groups, "critic", 1)
  storage = RolloutStorage("rl", 2, 2, obs, [3], "cpu")
  algorithm = HeftTeacherPPO(
    actor,
    critic,
    storage,
    device="cpu",
    num_learning_epochs=1,
    num_mini_batches=1,
    optimizer="muon",
    symmetry_cfg=None,
  )

  schedule = algorithm.step_schedule(1.0, 1)

  assert len(algorithm.optimizer.param_groups) == 4
  assert schedule["entropy_coef"] == 0.005
  assert algorithm.actor_learning_rate == 1.0e-4
  assert algorithm.critic_learning_rate == 5.0e-4
