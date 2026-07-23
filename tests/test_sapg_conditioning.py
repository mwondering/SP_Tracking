from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from rsl_rl.modules.distribution import GaussianDistribution

from sp_tracking.tasks.tracking.rl.sapg.conditioning import (
  BlockGaussianDistribution,
  PolicyConditionedLinear,
  PolicyContext,
)
from sp_tracking.tasks.tracking.rl.sapg.config import SAPGConfig
from sp_tracking.tasks.tracking.rl.sapg.update import (
  _entropy_loss,
  _on_policy_auxiliary_weight,
)


def test_policy_conditioned_linear_matches_explicit_concatenation() -> None:
  torch.manual_seed(4)
  context = PolicyContext(4)
  layer = PolicyConditionedLinear(nn.Linear(3, 5), context, 2)
  inputs = torch.randn(7, 3)
  policy_ids = torch.tensor([0, 1, 2, 3, 0, 1, 3])

  with context.use(policy_ids):
    actual = layer(inputs)
  combined_weight = torch.cat((layer.weight, layer.policy_weight), dim=1)
  combined_inputs = torch.cat((inputs, layer.policy_embedding[policy_ids]), dim=1)
  expected = nn.functional.linear(combined_inputs, combined_weight, layer.bias)

  torch.testing.assert_close(actual, expected)


def test_policy_context_defaults_to_last_block_and_repeats_for_symmetry() -> None:
  context = PolicyContext(4)
  assert context.resolve(3, torch.device("cpu")).tolist() == [3, 3, 3]

  with context.use(torch.tensor([0, 3])):
    assert context.resolve(4, torch.device("cpu")).tolist() == [0, 3, 0, 3]


def test_critic_reference_shares_embedding_without_registering_it_twice() -> None:
  context = PolicyContext(3)
  actor_layer = PolicyConditionedLinear(nn.Linear(2, 4), context, 5)
  critic_layer = PolicyConditionedLinear(
    nn.Linear(2, 4),
    context,
    5,
    policy_embedding=actor_layer.policy_embedding,
  )

  assert "policy_embedding" in dict(actor_layer.named_parameters())
  assert "policy_embedding" not in dict(critic_layer.named_parameters())
  with context.use(torch.tensor([0, 2])):
    (actor_layer(torch.ones(2, 2)) + critic_layer(torch.ones(2, 2))).sum().backward()
  assert actor_layer.policy_embedding.grad is not None


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_block_gaussian_selects_policy_std_and_defaults_to_leader(
  std_type: str,
) -> None:
  context = PolicyContext(3)
  source = GaussianDistribution(2, init_std=0.5, std_type=std_type)
  distribution = BlockGaussianDistribution(source, context)
  parameter = (
    distribution.std_param
    if std_type == "scalar"
    else distribution.log_std_param
  )
  with torch.no_grad():
    values = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.7, 0.8]])
    parameter.copy_(values if std_type == "scalar" else values.log())

  with context.use(torch.tensor([0, 2])):
    distribution.update(torch.zeros(2, 2))
  torch.testing.assert_close(distribution.std, values[[0, 2]])

  distribution.update(torch.zeros(1, 2))
  torch.testing.assert_close(distribution.std, values[[2]])


def test_official_entropy_coefficients_decrease_to_zero_for_leader() -> None:
  algorithm = SimpleNamespace(
    entropy_coef=9.0,
    _sapg_runtime=SimpleNamespace(
      config=SAPGConfig(enabled=True, exploration_type="entropy")
    ),
  )
  loss = _entropy_loss(
    algorithm,
    torch.ones(4),
    torch.tensor([0, 1, 2, 3]),
  )
  torch.testing.assert_close(loss, torch.tensor(0.25))


def test_auxiliary_subset_mean_is_weighted_as_a_masked_full_batch() -> None:
  samples, weight = _on_policy_auxiliary_weight(
    torch.tensor([True, False, True, False]), 4
  )

  assert samples == 2
  assert weight == 0.5


def test_config_normalizes_integer_valued_hydra_numbers() -> None:
  config = SAPGConfig.from_dict(
    {
      "enabled": True,
      "num_policy_blocks": 4.0,
      "local_parameter_dim": 8.0,
      "off_policy_ratio": 1.0,
      "value_eval_chunk_size": 16.0,
    }
  )

  assert type(config.num_policy_blocks) is int
  assert type(config.local_parameter_dim) is int
  assert type(config.off_policy_ratio) is int
  assert type(config.value_eval_chunk_size) is int


@pytest.mark.parametrize(
  ("override", "message"),
  [
    ({"enabled": "false"}, "enabled must be a boolean"),
    ({"method": "other"}, "method must be 'sapg' or 'cpo'"),
    ({"num_policy_blocks": 4.5}, "num_policy_blocks must be an integer"),
    ({"off_policy_ratio": "1"}, "off_policy_ratio must be an integer"),
    ({"entropy_coef_scale": float("nan")}, "must be a finite number"),
    ({"cpo_awac_temperature": 0.0}, "must be positive"),
    ({"cpo_awac_coef": -1.0}, "must be non-negative"),
  ],
)
def test_config_rejects_invalid_runtime_values(
  override: dict, message: str
) -> None:
  with pytest.raises((TypeError, ValueError), match=message):
    SAPGConfig.from_dict({"enabled": True, **override})
