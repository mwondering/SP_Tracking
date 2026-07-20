"""Observation-conditioned residual mixture-of-experts policy core."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
  """ONNX-friendly RMS normalization with a learned per-feature scale."""

  def __init__(self, hidden_dim: int, eps: float = 1.0e-6) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(int(hidden_dim)))
    self.eps = float(eps)

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
    return value * inverse_rms * self.weight


class LayerNormResidualBlock(nn.Module):
  """FlashSAC-shaped residual block using per-sample LayerNorm."""

  def __init__(self, hidden_dim: int, expansion: int = 4) -> None:
    super().__init__()
    hidden_dim = int(hidden_dim)
    expanded_dim = hidden_dim * int(expansion)
    if hidden_dim <= 0 or expanded_dim <= 0:
      raise ValueError("hidden_dim and expansion must be positive")

    # The following LayerNorm removes an affine bias from each hidden Linear,
    # so the bias is redundant.  Orthogonal initialization follows FlashSAC's
    # UnitLinear initialization without adopting its post-step projection.
    self.linear1 = nn.Linear(hidden_dim, expanded_dim, bias=False)
    self.norm1 = nn.LayerNorm(expanded_dim)
    self.linear2 = nn.Linear(expanded_dim, hidden_dim, bias=False)
    self.norm2 = nn.LayerNorm(hidden_dim)
    nn.init.orthogonal_(self.linear1.weight, gain=1.0)
    nn.init.orthogonal_(self.linear2.weight, gain=1.0)

  def residual(self, value: torch.Tensor) -> torch.Tensor:
    value = F.relu(self.norm1(self.linear1(value)))
    return F.relu(self.norm2(self.linear2(value)))

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    return value + self.residual(value)


class ObservationConditionedResidualMoE(nn.Module):
  """Dense-compute top-k residual MoE with a shared context backbone.

  All experts are evaluated in the first implementation.  Top-k sparsity is
  applied to the mixture probabilities, keeping the forward path simple and
  exportable while preserving the intended routing semantics.
  """

  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    *,
    context_hidden_dim: int = 1285,
    hidden_dim: int = 256,
    num_experts: int = 16,
    top_k: int = 8,
    expansion: int = 4,
    router_temperature: float = 1.0,
    router_init_std: float = 1.0e-2,
  ) -> None:
    super().__init__()
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    context_hidden_dim = int(context_hidden_dim)
    hidden_dim = int(hidden_dim)
    num_experts = int(num_experts)
    top_k = int(top_k)
    expansion = int(expansion)
    if min(input_dim, output_dim, context_hidden_dim, hidden_dim) <= 0:
      raise ValueError("MoE dimensions must be positive")
    if num_experts <= 1:
      raise ValueError("num_experts must be greater than one")
    if not 1 <= top_k <= num_experts:
      raise ValueError("top_k must be between one and num_experts")
    if expansion <= 0:
      raise ValueError("expansion must be positive")
    if router_temperature <= 0.0:
      raise ValueError("router_temperature must be positive")
    if router_init_std <= 0.0:
      raise ValueError("router_init_std must be positive")

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.context_hidden_dim = context_hidden_dim
    self.hidden_dim = hidden_dim
    self.num_experts = num_experts
    self.top_k = top_k
    self.expansion = expansion
    self.router_temperature = float(router_temperature)

    self.context_encoder = nn.Sequential(
      nn.Linear(input_dim, context_hidden_dim),
      nn.ReLU(),
      nn.Linear(context_hidden_dim, hidden_dim),
      nn.ReLU(),
    )
    self.shared_block = LayerNormResidualBlock(hidden_dim, expansion)
    self.router = nn.Linear(hidden_dim, num_experts, bias=False)
    self.experts = nn.ModuleList(
      LayerNormResidualBlock(hidden_dim, expansion)
      for _ in range(num_experts)
    )
    self.post_norm = RMSNorm(hidden_dim)
    self.output = nn.Linear(hidden_dim, output_dim)

    # Exactly-zero logits make torch.topk select the same experts for every
    # initial state.  A small random router keeps q close to uniform while
    # allowing the selected set to vary with observation content.
    nn.init.normal_(self.router.weight, mean=0.0, std=float(router_init_std))

  def _shared_features(self, value: torch.Tensor) -> torch.Tensor:
    return self.shared_block(self.context_encoder(value))

  def routing_probabilities_from_features(
    self, shared_features: torch.Tensor
  ) -> torch.Tensor:
    logits = self.router(shared_features) / self.router_temperature
    return torch.softmax(logits, dim=-1)

  def routing_probabilities(self, value: torch.Tensor) -> torch.Tensor:
    return self.routing_probabilities_from_features(
      self._shared_features(value)
    )

  def sparse_probabilities(
    self, dense_probabilities: torch.Tensor
  ) -> torch.Tensor:
    top_values, top_indices = torch.topk(
      dense_probabilities, self.top_k, dim=-1
    )
    top_values = top_values / top_values.sum(dim=-1, keepdim=True)
    return torch.zeros_like(dense_probabilities).scatter(
      -1, top_indices, top_values
    )

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    leading_shape = value.shape[:-1]
    flat_value = value.reshape(-1, value.shape[-1])
    shared = self._shared_features(flat_value)
    dense_probabilities = self.routing_probabilities_from_features(shared)
    sparse_probabilities = self.sparse_probabilities(dense_probabilities)

    expert_residuals = torch.stack(
      [expert.residual(shared) for expert in self.experts], dim=-2
    )
    mixed_residual = torch.sum(
      sparse_probabilities.unsqueeze(-1) * expert_residuals, dim=-2
    )
    mixed = self.post_norm(shared + mixed_residual)
    output = self.output(mixed)
    return output.reshape(*leading_shape, self.output_dim)

  @property
  def dense_parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self.parameters())

  @property
  def maximum_router_entropy(self) -> float:
    return math.log(self.num_experts)
