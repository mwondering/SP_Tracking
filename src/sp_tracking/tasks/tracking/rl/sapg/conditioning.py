"""Policy-local parameters and per-block Gaussian exploration for SAPG."""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import GaussianDistribution

from .config import SAPGConfig


class PolicyContext:
  """Transient target-policy IDs shared by conditioned actor and critic calls."""

  def __init__(self, num_policy_blocks: int) -> None:
    self.num_policy_blocks = int(num_policy_blocks)
    self.leader_policy_id = self.num_policy_blocks - 1
    self._stack: list[torch.Tensor | None] = []

  @contextmanager
  def use(self, policy_ids: torch.Tensor | None) -> Iterator[None]:
    if policy_ids is not None:
      if policy_ids.ndim != 1:
        raise ValueError("SAPG policy IDs must be a one-dimensional tensor")
      if policy_ids.numel() and (
        int(policy_ids.min()) < 0
        or int(policy_ids.max()) >= self.num_policy_blocks
      ):
        raise ValueError("SAPG policy ID is outside the configured block range")
    self._stack.append(policy_ids)
    try:
      yield
    finally:
      self._stack.pop()

  def resolve(self, batch_size: int, device: torch.device) -> torch.Tensor:
    active = self._stack[-1] if self._stack else None
    if active is None:
      return torch.full(
        (batch_size,), self.leader_policy_id, dtype=torch.long, device=device
      )
    ids = active.to(device=device, dtype=torch.long)
    if ids.numel() == batch_size:
      return ids
    if ids.numel() > 0 and batch_size % ids.numel() == 0:
      # RSL symmetry appends complete mirrored copies of the original batch.
      return ids.repeat(batch_size // ids.numel())
    raise ValueError(
      f"SAPG policy context contains {ids.numel()} IDs for batch size {batch_size}"
    )


class PolicyConditionedLinear(nn.Module):
  """A Linear layer plus the policy-local contribution used by SAPG.

  ``Linear(x) + W_policy @ phi[id]`` is exactly equivalent to applying one
  Linear layer to ``concat(x, phi[id])`` while preserving the original
  parameter names and shapes for checkpoint migration.
  """

  def __init__(
    self,
    linear: nn.Linear,
    context: PolicyContext,
    local_parameter_dim: int,
    *,
    policy_embedding: nn.Parameter | None = None,
  ) -> None:
    super().__init__()
    self.in_features = linear.in_features
    self.out_features = linear.out_features
    self.weight = linear.weight
    self.bias = linear.bias
    self.context = context
    self.policy_weight = nn.Parameter(
      torch.empty(
        linear.out_features,
        local_parameter_dim,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
      )
    )
    # Match the task model's already-initialized first-layer scale. This also
    # preserves HEFT's deliberately small orthogonal initialization.
    with torch.no_grad():
      scale = linear.weight.detach().std(unbiased=False).clamp_min(1.0e-8)
      nn.init.normal_(self.policy_weight, mean=0.0, std=float(scale))
    if policy_embedding is None:
      self.policy_embedding = nn.Parameter(
        torch.randn(
          context.num_policy_blocks,
          local_parameter_dim,
          device=linear.weight.device,
          dtype=linear.weight.dtype,
        )
      )
      object.__setattr__(self, "_policy_embedding_ref", None)
    else:
      # Do not register the shared actor embedding on the critic a second time.
      object.__setattr__(self, "_policy_embedding_ref", weakref.ref(policy_embedding))

  def _embedding(self) -> nn.Parameter:
    reference = self.__dict__.get("_policy_embedding_ref")
    if reference is None:
      return self.policy_embedding
    embedding = reference()
    if embedding is None:
      raise RuntimeError("The shared SAPG policy embedding is no longer alive")
    return embedding

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    policy_ids = self.context.resolve(value.shape[0], value.device)
    local = self._embedding()[policy_ids]
    return nn.functional.linear(value, self.weight, self.bias) + nn.functional.linear(
      local, self.policy_weight
    )

  def __deepcopy__(self, memo):
    """Fold the leader contribution into a plain Linear for deployment copies.

    Every actor export wrapper in this repository deep-copies its final MLP.
    Returning a normal Linear here keeps existing ONNX/JIT input contracts and
    removes all training-time policy context from the deployed leader policy.
    """
    if id(self) in memo:
      return memo[id(self)]
    folded = nn.Linear(
      self.in_features,
      self.out_features,
      bias=self.bias is not None,
      device=self.weight.device,
      dtype=self.weight.dtype,
    )
    with torch.no_grad():
      folded.weight.copy_(self.weight)
      if self.bias is not None:
        leader = self._embedding()[self.context.leader_policy_id]
        contribution = nn.functional.linear(leader, self.policy_weight)
        folded.bias.copy_(self.bias + contribution)
    memo[id(self)] = folded
    return folded


class BlockGaussianDistribution(GaussianDistribution):
  """State-independent Gaussian with one standard-deviation row per policy."""

  def __init__(
    self,
    source: GaussianDistribution,
    context: PolicyContext,
  ) -> None:
    nn.Module.__init__(self)
    self.output_dim = source.output_dim
    self.std_type = source.std_type
    self.std_range = list(source.std_range)
    self.log_std_range = list(source.log_std_range)
    self.context = context
    self._distribution = None
    if self.std_type == "scalar":
      initial = source.std_param.detach().repeat(context.num_policy_blocks, 1)
      self.std_param = nn.Parameter(
        initial, requires_grad=source.std_param.requires_grad
      )
    elif self.std_type == "log":
      initial = source.log_std_param.detach().repeat(context.num_policy_blocks, 1)
      self.log_std_param = nn.Parameter(
        initial, requires_grad=source.log_std_param.requires_grad
      )
    else:
      raise ValueError(f"Unsupported Gaussian std type for SAPG: {self.std_type}")

  def update(self, mlp_output: torch.Tensor) -> None:
    policy_ids = self.context.resolve(mlp_output.shape[0], mlp_output.device)
    if self.std_type == "scalar":
      table = self.std_param.clamp(self.std_range[0], self.std_range[1])
    else:
      log_table = self.log_std_param.clamp(
        self.log_std_range[0], self.log_std_range[1]
      )
      table = torch.exp(log_table)
    self._distribution = torch.distributions.Normal(mlp_output, table[policy_ids])


def _first_linear(mlp: nn.Module, model_name: str) -> tuple[str, nn.Linear]:
  for name, module in mlp.named_children():
    if isinstance(module, nn.Linear):
      return name, module
    break
  raise TypeError(
    f"SAPG requires {model_name}.mlp to start with torch.nn.Linear"
  )


def _replace_child(parent: nn.Module, name: str, child: nn.Module) -> None:
  if isinstance(parent, nn.Sequential):
    parent[int(name)] = child
  else:
    setattr(parent, name, child)


def install_policy_conditioning(
  actor: nn.Module,
  critic: nn.Module,
  config: SAPGConfig,
) -> PolicyContext:
  """Install SAPG-only modules before the algorithm creates its optimizers."""
  if getattr(actor, "is_recurrent", False) or getattr(critic, "is_recurrent", False):
    raise ValueError("SAPG currently supports feed-forward actor/critic models only")
  actor_mlp = getattr(actor, "mlp", None)
  critic_mlp = getattr(critic, "mlp", None)
  if not isinstance(actor_mlp, nn.Module) or not isinstance(critic_mlp, nn.Module):
    raise TypeError("SAPG requires actor and critic models exposing a final .mlp")
  distribution = getattr(actor, "distribution", None)
  if type(distribution) is not GaussianDistribution:
    raise TypeError("SAPG official mode requires GaussianDistribution")

  context = PolicyContext(config.num_policy_blocks)
  actor_name, actor_linear = _first_linear(actor_mlp, "actor")
  actor_conditioned = PolicyConditionedLinear(
    actor_linear,
    context,
    config.local_parameter_dim,
  )
  _replace_child(actor_mlp, actor_name, actor_conditioned)

  critic_name, critic_linear = _first_linear(critic_mlp, "critic")
  critic_conditioned = PolicyConditionedLinear(
    critic_linear,
    context,
    config.local_parameter_dim,
    policy_embedding=actor_conditioned.policy_embedding,
  )
  _replace_child(critic_mlp, critic_name, critic_conditioned)
  actor.distribution = BlockGaussianDistribution(distribution, context)

  # Plain attributes are intentionally non-module runtime handles.
  object.__setattr__(actor, "_sapg_policy_context", context)
  object.__setattr__(critic, "_sapg_policy_context", context)
  return context
