from __future__ import annotations

from types import SimpleNamespace

from hydra import compose, initialize_config_module
import torch
import torch.nn as nn
from tensordict import TensorDict

from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.rl.ppo import SPV51ContactEstimatorMoEPPO
from sp_tracking.tasks.tracking.rl.residual_moe import (
  LayerNormResidualBlock,
  ObservationConditionedResidualMoE,
)


def test_residual_moe_topk_probabilities_and_output_shape() -> None:
  torch.manual_seed(7)
  model = ObservationConditionedResidualMoE(
    11,
    3,
    context_hidden_dim=13,
    hidden_dim=8,
    num_experts=4,
    top_k=2,
    expansion=2,
  )
  value = torch.randn(5, 11)
  dense = model.routing_probabilities(value)
  sparse = model.sparse_probabilities(dense)

  assert model(value).shape == (5, 3)
  torch.testing.assert_close(dense.sum(dim=-1), torch.ones(5))
  torch.testing.assert_close(sparse.sum(dim=-1), torch.ones(5))
  assert torch.equal((sparse > 0.0).sum(dim=-1), torch.full((5,), 2))


def test_residual_blocks_use_bias_free_orthogonal_linears_and_layer_norm() -> None:
  block = LayerNormResidualBlock(hidden_dim=8, expansion=2)

  assert block.linear1.bias is None
  assert block.linear2.bias is None
  assert isinstance(block.norm1, nn.LayerNorm)
  assert isinstance(block.norm2, nn.LayerNorm)
  torch.testing.assert_close(
    block.linear1.weight.T @ block.linear1.weight,
    torch.eye(8),
    atol=1.0e-5,
    rtol=1.0e-5,
  )
  torch.testing.assert_close(
    block.linear2.weight @ block.linear2.weight.T,
    torch.eye(8),
    atol=1.0e-5,
    rtol=1.0e-5,
  )


def test_moe_action_head_starts_at_baseline_output_scale() -> None:
  model = ObservationConditionedResidualMoE(
    11,
    3,
    context_hidden_dim=13,
    hidden_dim=8,
    num_experts=4,
    top_k=2,
    expansion=2,
    output_init_gain=0.05,
  )

  torch.testing.assert_close(model.output.bias, torch.zeros(3))
  torch.testing.assert_close(
    model.output.weight @ model.output.weight.T,
    torch.eye(3) * 0.05**2,
    atol=1.0e-7,
    rtol=1.0e-5,
  )


def test_default_moe_policy_matches_30m_parameter_budget() -> None:
  model = ObservationConditionedResidualMoE(1651, 29)
  target_count = 30_000_000

  assert model.dense_parameter_count == 30_080_989
  assert abs(model.dense_parameter_count - target_count) / target_count < 3.0e-3


class _ToyRouter(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.linear = nn.Linear(2, 3, bias=False)

  def routing_probabilities(self, observations: TensorDict) -> torch.Tensor:
    return torch.softmax(self.linear(observations["router_input"]), dim=-1)


class _ActorWithIndependentEstimator(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.policy = nn.Linear(2, 2)
    self.estimator = nn.Linear(2, 2)


def test_estimator_gradient_clipping_is_independent_from_actor() -> None:
  actor = _ActorWithIndependentEstimator()
  algorithm = object.__new__(SPV51ContactEstimatorMoEPPO)
  algorithm.actor = actor
  algorithm._estimator_parameters = tuple(actor.estimator.parameters())
  algorithm._estimator_parameter_ids = {
    id(parameter) for parameter in algorithm._estimator_parameters
  }
  algorithm.estimator_max_grad_norm = 1.0
  for parameter in actor.parameters():
    parameter.grad = torch.full_like(parameter, 100.0)

  estimator_gradients_before = tuple(
    parameter.grad.clone() for parameter in algorithm._estimator_parameters
  )
  actor_parameters = tuple(
    algorithm._actor_parameters_for_gradient_clipping()
  )
  assert {id(parameter) for parameter in actor_parameters}.isdisjoint(
    algorithm._estimator_parameter_ids
  )
  nn.utils.clip_grad_norm_(actor_parameters, max_norm=1.0)
  for parameter, expected in zip(
    algorithm._estimator_parameters, estimator_gradients_before
  ):
    torch.testing.assert_close(parameter.grad, expected)

  algorithm._clip_auxiliary_gradients()
  estimator_norm = torch.linalg.vector_norm(
    torch.cat(
      [
        parameter.grad.flatten()
        for parameter in algorithm._estimator_parameters
      ]
    )
  )
  torch.testing.assert_close(estimator_norm, torch.tensor(1.0))


def test_collect_balance_loss_chunked_gradient_matches_full_rollout() -> None:
  actor = _ToyRouter()
  observations = TensorDict(
    {
      "router_input": torch.tensor(
        [
          [[1.0, 0.0], [0.0, 1.0]],
          [[1.0, 1.0], [-1.0, 0.5]],
        ]
      )
    },
    batch_size=[2, 2],
  )
  algorithm = object.__new__(SPV51ContactEstimatorMoEPPO)
  algorithm.actor = actor
  algorithm.storage = SimpleNamespace(step=2, observations=observations)
  algorithm.moe_collect_chunk_size = 3
  algorithm.moe_balance_loss_coef = 0.003
  algorithm.is_multi_gpu = False
  algorithm.gpu_world_size = 1
  algorithm._moe_balance_gradient = None
  algorithm._moe_balance_global_count = 0.0

  full_probabilities = actor.routing_probabilities(observations.flatten(0, 1))
  mean_probability = full_probabilities.mean(dim=0)
  expected_loss = 0.003 * (
    mean_probability * (mean_probability.log() + torch.log(torch.tensor(3.0)))
  ).sum()
  expected_gradient = torch.autograd.grad(
    expected_loss, actor.linear.weight
  )[0]

  metrics = algorithm._prepare_collect_auxiliary_loss()
  actor.zero_grad()
  algorithm._backward_collect_auxiliary_loss()

  assert "router_balance_kl" in metrics
  assert actor.linear.weight.grad is not None
  torch.testing.assert_close(
    actor.linear.weight.grad,
    expected_gradient,
    atol=1.0e-7,
    rtol=1.0e-5,
  )


def test_confidence_schedule_has_warmup_and_linear_ramp() -> None:
  algorithm = object.__new__(SPV51ContactEstimatorMoEPPO)
  algorithm.moe_confidence_loss_coef = 3.0e-4
  algorithm.moe_confidence_warmup_updates = 5000
  algorithm.moe_confidence_ramp_updates = 10000

  algorithm.moe_update_count = 4999
  assert algorithm._confidence_coefficient() == 0.0
  algorithm.moe_update_count = 10000
  assert algorithm._confidence_coefficient() == 1.5e-4
  algorithm.moe_update_count = 15000
  assert algorithm._confidence_coefficient() == 3.0e-4


def test_spv5_1_moe_task_exposes_closed_first_version_config() -> None:
  with initialize_config_module(
    version_base=None, config_module="sp_tracking.conf"
  ):
    cfg = compose(
      config_name="train",
      overrides=[
        "task=tracking_bfm_spv5_1_moe_actor_heft_critic_heft_reward"
      ],
    )
  prepared = prepare_train_cfg(cfg)

  assert prepared.agent.actor.class_name.endswith(
    ":SPV51ContactEstimatorMoEActor"
  )
  assert prepared.agent.actor.moe_num_experts == 16
  assert prepared.agent.actor.moe_top_k == 8
  assert prepared.agent.actor.moe_context_hidden_dim == 1280
  assert prepared.agent.actor.moe_hidden_dim == 448
  assert prepared.agent.actor.moe_router_temperature == 1.5
  assert prepared.agent.actor.moe_output_init_gain == 0.05
  assert prepared.agent.algorithm.class_name.endswith(
    ":SPV51ContactEstimatorMoEPPO"
  )
  assert prepared.agent.algorithm.moe_balance_loss_coef == 0.01
  assert prepared.agent.algorithm.moe_confidence_loss_coef == 0.0
  assert prepared.agent.algorithm.estimator_max_grad_norm == 1.0
