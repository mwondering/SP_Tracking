from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_module
import torch
import torch.nn as nn
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.mdp.multi_commands import (
  MultiMotionCommand,
  gradient_test_motion_assignment,
)
from sp_tracking.tasks.tracking.rl.policy_gradient_diagnostics import (
  PolicyGradientDiagnosticsPPO,
)
from sp_tracking.tasks.tracking.rl.runner import SpTrackingOnPolicyRunner


def _compose(*overrides: str):
  with initialize_config_module(
    version_base=None, config_module="sp_tracking.conf"
  ):
    return compose(config_name="train", overrides=list(overrides))


def test_gradient_task_config_is_spv5_small_network_and_rollout_only() -> None:
  cfg = _compose("task=test_policy_gradients")
  prepared = prepare_train_cfg(cfg)
  env = build_env_cfg(cfg.task)

  assert cfg.task.name == "test_policy_gradients"
  assert env.scene.num_envs == 16384
  assert prepared.agent.max_iterations == 300000
  assert prepared.agent.actor.hidden_dims == (512, 256, 128)
  assert prepared.agent.critic.hidden_dims == (512, 256, 128)
  assert prepared.agent.algorithm.num_mini_batches == 16
  assert prepared.agent.algorithm.class_name.endswith(
    ":PolicyGradientDiagnosticsPPO"
  )
  assert set(env.observations) >= {
    "gradient_motion_label",
    "gradient_motion_phase",
  }
  assert "gradient_motion_label" not in prepared.agent.obs_groups["actor"]
  assert "gradient_motion_label" not in prepared.agent.obs_groups["critic"]
  assert env.commands["motion"].gradient_test_mode == "mixed"
  assert env.commands["motion"].rewind.enabled is False


def test_motion_assignment_keeps_semantic_labels_across_three_runs() -> None:
  env_ids = torch.arange(8)

  simple_index, simple_label = gradient_test_motion_assignment(
    "simple", env_ids, 8
  )
  hard_index, hard_label = gradient_test_motion_assignment("hard", env_ids, 8)
  mixed_index, mixed_label = gradient_test_motion_assignment(
    "mixed", env_ids, 8
  )

  torch.testing.assert_close(simple_index, torch.zeros(8, dtype=torch.long))
  torch.testing.assert_close(simple_label, torch.zeros(8, dtype=torch.long))
  torch.testing.assert_close(hard_index, torch.zeros(8, dtype=torch.long))
  torch.testing.assert_close(hard_label, torch.ones(8, dtype=torch.long))
  torch.testing.assert_close(
    mixed_index, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
  )
  torch.testing.assert_close(mixed_label, mixed_index)


def test_mixed_motion_assignment_rejects_odd_per_rank_env_count() -> None:
  try:
    gradient_test_motion_assignment("mixed", torch.arange(3), 3)
  except ValueError as error:
    assert "even number" in str(error)
  else:
    raise AssertionError("odd mixed environment count was accepted")


def test_gradient_task_resolves_two_explicit_npz_files_without_sharding(
  tmp_path: Path,
) -> None:
  simple = tmp_path / "simple.npz"
  hard = tmp_path / "hard.npz"
  simple.touch()
  hard.touch()
  command = object.__new__(MultiMotionCommand)
  command.cfg = SimpleNamespace(
    gradient_test_mode="mixed",
    gradient_test_simple_motion_file=str(simple),
    gradient_test_hard_motion_file=str(hard),
  )

  assert command._resolve_motion_files() == [str(simple), str(hard)]

  command.cfg.gradient_test_mode = "simple"
  command.cfg.gradient_test_hard_motion_file = ""
  assert command._resolve_motion_files() == [str(simple)]
  command.cfg.gradient_test_mode = "hard"
  command.cfg.gradient_test_simple_motion_file = ""
  command.cfg.gradient_test_hard_motion_file = str(hard)
  assert command._resolve_motion_files() == [str(hard)]


def _diagnostic_storage() -> RolloutStorage:
  num_envs = 4
  num_steps = 2
  observation = TensorDict(
    {
      "gradient_motion_label": torch.tensor([[0], [0], [1], [1]]),
      "gradient_motion_phase": torch.zeros(num_envs, 1),
      "sample_id": torch.arange(num_envs).unsqueeze(-1),
    },
    batch_size=[num_envs],
  )
  storage = RolloutStorage("rl", num_envs, num_steps, observation, [1], "cpu")
  for step in range(num_steps):
    storage.observations[step] = observation.clone()
    storage.observations[step]["sample_id"] += step * num_envs
  storage.distribution_params = (
    torch.zeros(num_steps, num_envs, 1),
    torch.ones(num_steps, num_envs, 1),
  )
  return storage


def test_stratified_generator_puts_both_motions_in_every_minibatch() -> None:
  harness = SimpleNamespace(
    storage=_diagnostic_storage(),
    gradient_stratified_minibatches=True,
    gradient_motion_label_group="gradient_motion_label",
    num_mini_batches=2,
    num_learning_epochs=2,
  )

  batches = list(
    PolicyGradientDiagnosticsPPO._policy_gradient_mini_batch_generator(harness)
  )

  assert len(batches) == 4
  for batch in batches:
    labels = batch.observations["gradient_motion_label"].reshape(-1)
    assert int((labels == 0).sum()) == 2
    assert int((labels == 1).sum()) == 2
  first_epoch_ids = torch.cat(
    [batch.observations["sample_id"].reshape(-1) for batch in batches[:2]]
  )
  torch.testing.assert_close(
    first_epoch_ids.sort().values, torch.arange(8)
  )


def test_diagnostic_generator_can_delegate_to_standard_shuffling() -> None:
  harness = SimpleNamespace(
    storage=_diagnostic_storage(),
    gradient_stratified_minibatches=False,
    num_mini_batches=2,
    num_learning_epochs=1,
  )

  batches = list(
    PolicyGradientDiagnosticsPPO._policy_gradient_mini_batch_generator(harness)
  )

  assert len(batches) == 2
  assert sum(batch.actions.shape[0] for batch in batches) == 8


def test_gradient_hook_records_actor_and_critic_geometry() -> None:
  torch.manual_seed(7)
  algorithm = object.__new__(PolicyGradientDiagnosticsPPO)
  actor = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
  critic = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
  algorithm._gradient_actor_parameters = tuple(actor.parameters())
  algorithm._gradient_critic_parameters = tuple(critic.parameters())
  algorithm.gradient_motion_label_group = "gradient_motion_label"
  algorithm.gradient_motion_phase_group = "gradient_motion_phase"
  algorithm.gradient_diagnostics_eps = 1.0e-12
  algorithm.num_mini_batches = 1
  algorithm.clip_param = 0.2
  algorithm.entropy_coef = 0.005
  algorithm.value_loss_coef = 1.0
  algorithm.is_multi_gpu = False
  algorithm.gpu_global_rank = 0
  algorithm._gradient_global_update_step = 0
  algorithm._gradient_diagnostic_records = []

  features = torch.tensor(
    [[1.0, 0.0], [0.5, 1.0], [-1.0, 0.5], [0.0, -1.0]]
  )
  actor_weights = torch.tensor([1.0, -0.5, 1.5, -1.0])
  targets = torch.tensor([0.5, -0.25, 1.0, -1.0])
  surrogate_terms = actor(features).reshape(-1) * actor_weights
  value_terms = (critic(features).reshape(-1) - targets).square()
  old_log_prob = torch.zeros(4, 1)
  new_log_prob = torch.tensor([0.02, -0.01, 0.04, -0.03]).unsqueeze(-1)
  ratio = new_log_prob.exp()
  observations = TensorDict(
    {
      "gradient_motion_label": torch.tensor([[0], [0], [1], [1]]),
      "gradient_motion_phase": torch.tensor([[0.1], [0.2], [0.7], [0.8]]),
    },
    batch_size=[4],
  )
  batch = RolloutStorage.Batch(
    observations=observations,
    advantages=torch.tensor([[1.0], [0.5], [-0.5], [-1.0]]),
    returns=targets.unsqueeze(-1),
    old_actions_log_prob=old_log_prob,
  )

  algorithm._diagnose_policy_gradient_batch(
    batch=batch,
    surrogate_terms=surrogate_terms,
    value_terms=value_terms,
    actions_log_prob=new_log_prob,
    ratio=ratio,
    entropy=torch.ones(4),
    update_index=0,
  )
  records = algorithm.drain_gradient_diagnostics()

  assert len(records) == 1
  record = records[0]
  assert record["simple_sample_count"] == 2
  assert record["hard_sample_count"] == 2
  assert record["actor_simple_grad_norm"] > 0.0
  assert record["actor_hard_grad_norm"] > 0.0
  assert record["critic_simple_grad_norm"] > 0.0
  assert record["critic_hard_grad_norm"] > 0.0
  assert -1.0 <= record["actor_grad_cosine"] <= 1.0
  assert -1.0 <= record["critic_grad_cosine"] <= 1.0
  assert record["simple_clip_fraction"] == 0.0
  assert record["hard_clip_fraction"] == 0.0
  assert algorithm.drain_gradient_diagnostics() == []


class _ScalarWriter:
  def __init__(self) -> None:
    self.scalars: list[tuple[str, float, int]] = []

  def add_scalar(self, tag: str, value: float, iteration: int) -> None:
    self.scalars.append((tag, value, iteration))


def test_runner_persists_raw_minibatches_and_logs_iteration_aggregates(
  tmp_path: Path,
) -> None:
  records = [
    {
      "global_update_step": index,
      "learning_epoch": 0,
      "minibatch_index": index,
      "simple_sample_count": 4,
      "hard_sample_count": 4,
      "actor_simple_grad_norm": 1.0 + index,
      "actor_hard_grad_norm": 3.0 + index,
      "actor_grad_cosine": -0.5 + index,
    }
    for index in range(2)
  ]
  algorithm = SimpleNamespace(
    drain_gradient_diagnostics=lambda: records
  )
  writer = _ScalarWriter()
  runner = object.__new__(SpTrackingOnPolicyRunner)
  runner.alg = algorithm
  runner.gpu_global_rank = 0
  runner.logger = SimpleNamespace(log_dir=str(tmp_path), writer=writer)
  runner._gradient_diagnostics_db = None

  summary = runner._flush_policy_gradient_diagnostics(iteration=12)
  runner._close_gradient_diagnostics()

  assert summary["actor_simple_grad_norm"] == {
    "mean": 1.5,
    "std": 0.5,
    "min": 1.0,
    "max": 2.0,
  }
  assert any(
    tag == "GradientDiagnostics/actor_grad_cosine/mean"
    for tag, _, _ in writer.scalars
  )
  connection = sqlite3.connect(tmp_path / "gradient_diagnostics.sqlite")
  try:
    rows = connection.execute(
      "SELECT iteration, global_update_step, actor_simple_grad_norm "
      "FROM minibatch_gradients ORDER BY global_update_step"
    ).fetchall()
  finally:
    connection.close()
  assert rows == [(12, 0, 1.0), (12, 1, 2.0)]
