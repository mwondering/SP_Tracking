"""Two-motion PPO gradient diagnostics for the SPV5 experiment task."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from rsl_rl.storage import RolloutStorage

from .ppo import SPV5ReferenceEncoderPPO


MOTION_LABELS = ((0, "simple"), (1, "hard"))


def _finite_float(value: torch.Tensor | float) -> float | None:
  scalar = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
  return scalar if math.isfinite(scalar) else None


def _cosine(first: torch.Tensor, second: torch.Tensor, eps: float) -> float | None:
  first_norm = torch.linalg.vector_norm(first)
  second_norm = torch.linalg.vector_norm(second)
  if float(first_norm.item()) <= eps or float(second_norm.item()) <= eps:
    return None
  return _finite_float(torch.dot(first, second) / (first_norm * second_norm))


class PolicyGradientDiagnosticsPPO(SPV5ReferenceEncoderPPO):
  """SPV5 PPO that observes exact per-motion minibatch gradient geometry."""

  _STATE_KEY = "policy_gradient_diagnostics_state"

  def __init__(
    self,
    *args,
    gradient_motion_label_group: str = "gradient_motion_label",
    gradient_motion_phase_group: str = "gradient_motion_phase",
    gradient_stratified_minibatches: bool = True,
    gradient_diagnostics_eps: float = 1.0e-12,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if hasattr(self, "_sapg_runtime"):
      raise ValueError("PolicyGradientDiagnosticsPPO does not support SAPG")
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError("PolicyGradientDiagnosticsPPO requires feed-forward models")
    self.gradient_motion_label_group = str(gradient_motion_label_group)
    self.gradient_motion_phase_group = str(gradient_motion_phase_group)
    self.gradient_stratified_minibatches = bool(
      gradient_stratified_minibatches
    )
    self.gradient_diagnostics_eps = float(gradient_diagnostics_eps)
    if self.gradient_diagnostics_eps <= 0.0:
      raise ValueError("gradient_diagnostics_eps must be positive")
    policy_mlp = getattr(self.actor, "mlp", None)
    if not isinstance(policy_mlp, nn.Module):
      raise TypeError("PolicyGradientDiagnosticsPPO requires actor.mlp")
    distribution = getattr(self.actor, "distribution", None)
    if not isinstance(distribution, nn.Module):
      raise TypeError("PolicyGradientDiagnosticsPPO requires actor.distribution")
    actor_parameters = (*policy_mlp.parameters(), *distribution.parameters())
    self._gradient_actor_parameters = tuple(
      dict.fromkeys(
        parameter for parameter in actor_parameters if parameter.requires_grad
      )
    )
    self._gradient_critic_parameters = tuple(
      parameter for parameter in self.critic.parameters() if parameter.requires_grad
    )
    if not self._gradient_actor_parameters or not self._gradient_critic_parameters:
      raise ValueError("gradient diagnostics found an empty parameter set")
    self._gradient_diagnostic_records: list[dict[str, int | float | None]] = []
    self._gradient_global_update_step = 0

  def _policy_gradient_mini_batch_generator(self):
    if not self.gradient_stratified_minibatches:
      yield from self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
      return

    storage = self.storage
    observations = storage.observations.flatten(0, 1)
    labels = observations[self.gradient_motion_label_group].reshape(-1).long()
    unique_labels = sorted(int(value) for value in torch.unique(labels).tolist())
    if not unique_labels or any(label not in {0, 1} for label in unique_labels):
      raise ValueError(
        "gradient motion labels must contain only 0=simple and/or 1=hard, "
        f"got {unique_labels}"
      )

    per_label_indices: dict[int, torch.Tensor] = {}
    for label in unique_labels:
      indices = torch.where(labels == label)[0]
      if indices.numel() % self.num_mini_batches:
        raise ValueError(
          f"motion label {label} has {indices.numel()} rollout samples, which "
          f"is not divisible by num_mini_batches={self.num_mini_batches}"
        )
      order = torch.randperm(indices.numel(), device=indices.device)
      per_label_indices[label] = indices[order]

    batches: list[torch.Tensor] = []
    for minibatch_index in range(self.num_mini_batches):
      pieces = []
      for label in unique_labels:
        indices = per_label_indices[label]
        per_batch = indices.numel() // self.num_mini_batches
        start = minibatch_index * per_batch
        pieces.append(indices[start : start + per_batch])
      batch_indices = torch.cat(pieces)
      shuffle = torch.randperm(batch_indices.numel(), device=batch_indices.device)
      batches.append(batch_indices[shuffle])

    actions = storage.actions.flatten(0, 1)
    values = storage.values.flatten(0, 1)
    returns = storage.returns.flatten(0, 1)
    advantages = storage.advantages.flatten(0, 1)
    old_actions_log_prob = storage.actions_log_prob.flatten(0, 1)
    if storage.distribution_params is None:
      raise RuntimeError("rollout storage has no behavior distribution parameters")
    old_distribution_params = tuple(
      parameter.flatten(0, 1) for parameter in storage.distribution_params
    )

    for _ in range(self.num_learning_epochs):
      for batch_indices in batches:
        yield RolloutStorage.Batch(
          observations=observations[batch_indices],
          actions=actions[batch_indices],
          values=values[batch_indices],
          advantages=advantages[batch_indices],
          returns=returns[batch_indices],
          old_actions_log_prob=old_actions_log_prob[batch_indices],
          old_distribution_params=tuple(
            parameter[batch_indices] for parameter in old_distribution_params
          ),
        )

  def _global_label_counts(self, labels: torch.Tensor) -> tuple[int, int]:
    counts = torch.bincount(labels, minlength=2).to(dtype=torch.float64)
    if self.is_multi_gpu:
      dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return int(counts[0].item()), int(counts[1].item())

  def _global_metric_moments(
    self, metrics: dict[str, torch.Tensor], mask: torch.Tensor
  ) -> dict[str, tuple[float | None, float | None]]:
    packed_values: list[torch.Tensor] = []
    for values in metrics.values():
      selected = values[mask].reshape(-1).detach().double()
      packed_values.extend(
        (
          selected.sum(),
          selected.square().sum(),
          selected.new_tensor(float(selected.numel())),
        )
      )
    packed = torch.stack(packed_values)
    if self.is_multi_gpu:
      dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    results: dict[str, tuple[float | None, float | None]] = {}
    for index, name in enumerate(metrics):
      total, square_total, count_tensor = packed[index * 3 : index * 3 + 3]
      count = float(count_tensor.item())
      if count <= 0.0:
        results[name] = (None, None)
        continue
      mean = total / count
      variance = (square_total / count - mean.square()).clamp_min(0.0)
      results[name] = (_finite_float(mean), _finite_float(variance.sqrt()))
    return results

  def _global_task_gradient(
    self,
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    local_count: int,
    global_count: int,
  ) -> torch.Tensor:
    parameter_tuple = tuple(parameters)
    gradients = torch.autograd.grad(
      loss,
      parameter_tuple,
      retain_graph=True,
      allow_unused=True,
    )
    flat = torch.cat(
      tuple(
        (
          gradient.detach().reshape(-1)
          if gradient is not None
          else parameter.detach().new_zeros(parameter.numel())
        )
        for parameter, gradient in zip(parameter_tuple, gradients)
      )
    )
    weighted = flat * float(local_count)
    if self.is_multi_gpu:
      dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
    if global_count <= 0:
      raise RuntimeError("cannot compute a gradient from an empty motion slice")
    return weighted / float(global_count)

  def _record_pair_geometry(
    self,
    record: dict[str, int | float | None],
    prefix: str,
    gradients: dict[int, torch.Tensor],
  ) -> None:
    for label, name in MOTION_LABELS:
      gradient = gradients.get(label)
      record[f"{prefix}_{name}_grad_norm"] = (
        _finite_float(torch.linalg.vector_norm(gradient))
        if gradient is not None
        else None
      )
    if set(gradients) != {0, 1}:
      for suffix in (
        "grad_norm_ratio_simple_over_hard",
        "grad_dot",
        "grad_cosine",
        "grad_cancellation",
        "simple_aggregate_cosine",
        "hard_aggregate_cosine",
      ):
        record[f"{prefix}_{suffix}"] = None
      return

    simple = gradients[0]
    hard = gradients[1]
    simple_norm = torch.linalg.vector_norm(simple)
    hard_norm = torch.linalg.vector_norm(hard)
    record[f"{prefix}_grad_norm_ratio_simple_over_hard"] = (
      _finite_float(simple_norm / hard_norm)
      if float(hard_norm.item()) > self.gradient_diagnostics_eps
      else None
    )
    record[f"{prefix}_grad_dot"] = _finite_float(torch.dot(simple, hard))
    record[f"{prefix}_grad_cosine"] = _cosine(
      simple, hard, self.gradient_diagnostics_eps
    )
    norm_sum = simple_norm + hard_norm
    record[f"{prefix}_grad_cancellation"] = (
      _finite_float(torch.linalg.vector_norm(simple + hard) / norm_sum)
      if float(norm_sum.item()) > self.gradient_diagnostics_eps
      else None
    )
    aggregate = 0.5 * (simple + hard)
    record[f"{prefix}_simple_aggregate_cosine"] = _cosine(
      simple, aggregate, self.gradient_diagnostics_eps
    )
    record[f"{prefix}_hard_aggregate_cosine"] = _cosine(
      hard, aggregate, self.gradient_diagnostics_eps
    )

  def _diagnose_policy_gradient_batch(
    self,
    *,
    batch: RolloutStorage.Batch,
    surrogate_terms: torch.Tensor,
    value_terms: torch.Tensor,
    actions_log_prob: torch.Tensor,
    ratio: torch.Tensor,
    entropy: torch.Tensor,
    update_index: int,
  ) -> None:
    observations = batch.observations
    if observations is None:
      raise RuntimeError("gradient diagnostics received an empty observation batch")
    labels = observations[self.gradient_motion_label_group].reshape(-1).long()
    phase = observations[self.gradient_motion_phase_group].reshape(-1)
    unique_labels = sorted(int(value) for value in torch.unique(labels).tolist())
    if not unique_labels or any(label not in {0, 1} for label in unique_labels):
      raise ValueError(f"unexpected gradient motion labels: {unique_labels}")

    actor_gradients: dict[int, torch.Tensor] = {}
    critic_gradients: dict[int, torch.Tensor] = {}
    record: dict[str, int | float | None] = {
      "global_update_step": self._gradient_global_update_step,
      "learning_epoch": update_index // self.num_mini_batches,
      "minibatch_index": update_index % self.num_mini_batches,
    }
    advantages = batch.advantages.reshape(-1)  # type: ignore[union-attr]
    returns = batch.returns.reshape(-1)  # type: ignore[union-attr]
    old_log_prob = batch.old_actions_log_prob.reshape(-1)  # type: ignore[union-attr]
    new_log_prob = actions_log_prob.reshape(-1)
    log_ratio = new_log_prob - old_log_prob
    approximate_kl = (ratio.reshape(-1) - 1.0) - log_ratio
    clip_fraction = (ratio.reshape(-1) - 1.0).abs() > self.clip_param
    global_counts = self._global_label_counts(labels)

    for label, name in MOTION_LABELS:
      mask = labels == label
      local_count = int(mask.sum().item())
      global_count = global_counts[label]
      record[f"{name}_sample_count"] = global_count
      if global_count == 0:
        for metric in (
          "surrogate_mean",
          "value_loss_mean",
          "advantage_mean",
          "advantage_std",
          "return_mean",
          "return_std",
          "entropy_mean",
          "approx_kl_mean",
          "clip_fraction",
          "phase_mean",
        ):
          record[f"{name}_{metric}"] = None
        continue
      if local_count == 0:
        raise RuntimeError(
          "Every distributed rank must contain both motions in mixed mode"
        )

      actor_loss = (
        surrogate_terms[mask] - self.entropy_coef * entropy[mask]
      ).mean()
      critic_loss = self.value_loss_coef * value_terms[mask].mean()
      actor_gradient = self._global_task_gradient(
        actor_loss, self._gradient_actor_parameters, local_count, global_count
      )
      critic_gradient = self._global_task_gradient(
        critic_loss, self._gradient_critic_parameters, local_count, global_count
      )
      actor_gradients[label] = actor_gradient
      critic_gradients[label] = critic_gradient

      moments = self._global_metric_moments(
        {
          "surrogate": surrogate_terms,
          "value_loss": value_terms,
          "advantage": advantages,
          "return": returns,
          "entropy": entropy,
          "approx_kl": approximate_kl,
          "clip_fraction": clip_fraction.to(torch.float32),
          "phase": phase,
        },
        mask,
      )
      mean_metrics = {
        "surrogate_mean": "surrogate",
        "value_loss_mean": "value_loss",
        "entropy_mean": "entropy",
        "approx_kl_mean": "approx_kl",
        "clip_fraction": "clip_fraction",
        "phase_mean": "phase",
      }
      for output_name, metric in mean_metrics.items():
        record[f"{name}_{output_name}"] = moments[metric][0]
      record[f"{name}_advantage_mean"], record[f"{name}_advantage_std"] = (
        moments["advantage"]
      )
      record[f"{name}_return_mean"], record[f"{name}_return_std"] = moments[
        "return"
      ]

    self._record_pair_geometry(record, "actor", actor_gradients)
    self._record_pair_geometry(record, "critic", critic_gradients)
    if self.gpu_global_rank == 0:
      self._gradient_diagnostic_records.append(record)
    self._gradient_global_update_step += 1

  def drain_gradient_diagnostics(self) -> list[dict[str, int | float | None]]:
    records = self._gradient_diagnostic_records
    self._gradient_diagnostic_records = []
    return records

  def save(self) -> dict:
    state = super().save()
    state[self._STATE_KEY] = {
      "global_update_step": self._gradient_global_update_step,
    }
    return state

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    state = loaded_dict.get(self._STATE_KEY, {})
    self._gradient_global_update_step = int(
      state.get("global_update_step", self._gradient_global_update_step)
    )
    return load_iteration
