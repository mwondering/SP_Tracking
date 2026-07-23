"""Trajectory-preserving SAPG aggregation over RSL-RL rollout storage."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

import torch
from rsl_rl.storage import RolloutStorage


def rollout_policy_ids(
  num_envs: int,
  num_policy_blocks: int,
  device: torch.device | str,
  *,
  require_divisible: bool = True,
) -> torch.Tensor:
  """Return official contiguous block IDs, with the last block as leader."""
  if num_envs % num_policy_blocks:
    if require_divisible:
      raise ValueError(
        f"SAPG training requires num_envs ({num_envs}) to be divisible by "
        f"num_policy_blocks ({num_policy_blocks})"
      )
    return torch.full(
      (num_envs,), num_policy_blocks - 1, dtype=torch.long, device=device
    )
  block_size = num_envs // num_policy_blocks
  return torch.arange(num_policy_blocks, device=device).repeat_interleave(
    block_size
  )


@dataclass
class SAPGAggregatedData:
  """Index-only aggregate plus SAPG-specific value/return targets."""

  source_indices: torch.Tensor
  source_policy_ids: torch.Tensor
  target_policy_ids: torch.Tensor
  off_policy_mask: torch.Tensor
  values: torch.Tensor
  returns: torch.Tensor
  advantages: torch.Tensor
  kl_reference_params: tuple[torch.Tensor, ...]
  selected_follower_ids: torch.Tensor
  num_trajectories: int
  horizon: int

  @property
  def num_samples(self) -> int:
    return int(self.source_indices.numel())

  def update_kl_reference(
    self,
    aggregate_indices: torch.Tensor,
    distribution_params: tuple[torch.Tensor, ...],
  ) -> None:
    """Persist current policy parameters for the next PPO learning epoch.

    Official SAPG updates the dataset's ``mu`` and ``sigma`` after every
    mini-batch. These mutable references are used only by adaptive-KL
    scheduling; rollout actions, log probabilities, and behavior-policy
    parameters remain immutable.
    """
    if len(distribution_params) != len(self.kl_reference_params):
      raise ValueError(
        "SAPG KL reference parameter count does not match the actor distribution"
      )
    indices = aggregate_indices.to(
      device=self.source_indices.device, dtype=torch.long
    )
    if indices.ndim != 1:
      raise ValueError("SAPG aggregate indices must be one-dimensional")
    for reference, current in zip(
      self.kl_reference_params, distribution_params, strict=True
    ):
      expected_shape = (indices.numel(), *reference.shape[1:])
      if tuple(current.shape) != expected_shape:
        raise ValueError(
          "SAPG current distribution parameter has shape "
          f"{tuple(current.shape)}, expected {expected_shape}"
        )
      reference.index_copy_(
        0,
        indices,
        current.detach().to(device=reference.device, dtype=reference.dtype),
      )


def build_aggregated_data(
  storage: RolloutStorage,
  selected_follower_ids: torch.Tensor,
  leader_values: torch.Tensor,
  leader_next_values: torch.Tensor,
  *,
  num_policy_blocks: int,
  gamma: float,
) -> SAPGAggregatedData:
  """Build the official leader-follower aggregate without copying observations.

  ``leader_values`` and ``leader_next_values`` have shape ``[H, R*B, 1]``
  and correspond to the selected follower environments in follower-ID order.
  """
  horizon = int(storage.num_transitions_per_env)
  num_envs = int(storage.num_envs)
  device = storage.values.device
  policy_ids = rollout_policy_ids(num_envs, num_policy_blocks, device)
  block_size = num_envs // num_policy_blocks
  selected_follower_ids = selected_follower_ids.to(device=device, dtype=torch.long)
  off_envs = torch.cat(
    [
      torch.arange(
        int(follower) * block_size,
        (int(follower) + 1) * block_size,
        device=device,
      )
      for follower in selected_follower_ids
    ]
  )
  expected_shape = (horizon, off_envs.numel(), 1)
  if tuple(leader_values.shape) != expected_shape:
    raise ValueError(
      f"leader_values has shape {tuple(leader_values.shape)}, expected {expected_shape}"
    )
  if tuple(leader_next_values.shape) != expected_shape:
    raise ValueError(
      "leader_next_values has shape "
      f"{tuple(leader_next_values.shape)}, expected {expected_shape}"
    )

  # Official SAPG converts rollout data to env-major trajectories before its
  # trajectory-level shuffle. Map those descriptors back to RSL's time-major
  # flattened storage only when a mini-batch is gathered.
  source_envs = torch.cat((torch.arange(num_envs, device=device), off_envs))
  source_indices = (
    torch.arange(horizon, device=device).unsqueeze(1) * num_envs
    + source_envs.unsqueeze(0)
  ).transpose(0, 1).reshape(-1)

  source_policy_ids = policy_ids[source_envs].repeat_interleave(horizon)
  target_policy_ids = source_policy_ids.clone()
  on_policy_samples = num_envs * horizon
  target_policy_ids[on_policy_samples:] = num_policy_blocks - 1
  off_policy_mask = torch.zeros(
    source_indices.numel(), dtype=torch.bool, device=device
  )
  off_policy_mask[on_policy_samples:] = True

  on_values = storage.values.transpose(0, 1).reshape(-1, 1)
  on_returns = storage.returns.transpose(0, 1).reshape(-1, 1)
  off_values = leader_values.transpose(0, 1).reshape(-1, 1)
  alive = 1.0 - storage.dones[:, off_envs].to(storage.rewards.dtype)
  off_returns = storage.rewards[:, off_envs] + gamma * alive * leader_next_values
  off_returns = off_returns.transpose(0, 1).reshape(-1, 1)
  values = torch.cat((on_values, off_values))
  returns = torch.cat((on_returns, off_returns))
  advantages = returns - values
  flat_distribution_params = tuple(
    parameter.flatten(0, 1) for parameter in storage.distribution_params
  )
  kl_reference_params = tuple(
    parameter[source_indices].detach().clone()
    for parameter in flat_distribution_params
  )

  return SAPGAggregatedData(
    source_indices=source_indices,
    source_policy_ids=source_policy_ids,
    target_policy_ids=target_policy_ids,
    off_policy_mask=off_policy_mask,
    values=values,
    returns=returns,
    advantages=advantages,
    kl_reference_params=kl_reference_params,
    selected_follower_ids=selected_follower_ids,
    num_trajectories=int(source_envs.numel()),
    horizon=horizon,
  )


def normalize_aggregated_advantages(
  data: SAPGAggregatedData,
  storage: RolloutStorage,
  *,
  is_multi_gpu: bool,
) -> None:
  """Normalize after aggregation, using HEFT's valid-state mask when present."""
  flat_obs = storage.observations.flatten(0, 1)
  if "is_init" in flat_obs.keys():
    valid = (~flat_obs["is_init"][data.source_indices]).to(
      data.advantages.dtype
    )
  else:
    valid = torch.ones_like(data.advantages)
  total = (data.advantages * valid).sum()
  square_total = (data.advantages.square() * valid).sum()
  count = valid.sum().clamp_min(1.0)
  if is_multi_gpu:
    packed = torch.stack((total, square_total, count))
    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    total, square_total, count = packed
  mean = total / count
  variance = (square_total / count - mean.square()).clamp_min(0.0)
  data.advantages = (data.advantages - mean) / (variance.sqrt() + 1.0e-8)


def sapg_mini_batch_generator(
  storage: RolloutStorage,
  data: SAPGAggregatedData,
  *,
  num_mini_batches: int,
  num_epochs: int,
) -> Generator[RolloutStorage.Batch, None, None]:
  """Yield official-style mini-batches after one trajectory-preserving shuffle.

  The official ``rl_games`` dataset uses floor division to choose the number
  of mini-batches and appends any remainder to the final batch.  Keeping that
  detail matters when SAPG ablations make the aggregate size non-divisible by
  the base PPO mini-batch size: emitting an extra short batch would add an
  optimizer step and overweight the remainder.
  """
  if num_mini_batches <= 0:
    raise ValueError("SAPG num_mini_batches must be positive")
  base_batch_size = storage.num_envs * storage.num_transitions_per_env
  mini_batch_size = base_batch_size // num_mini_batches
  if mini_batch_size <= 0:
    raise ValueError("SAPG mini-batch size must be positive")
  batches_per_epoch = data.num_samples // mini_batch_size
  if batches_per_epoch <= 0:
    raise ValueError("SAPG aggregate is smaller than one mini-batch")

  device = data.source_indices.device
  trajectory_order = torch.randperm(data.num_trajectories, device=device)
  trajectory_samples = (
    trajectory_order.unsqueeze(1) * data.horizon
    + torch.arange(data.horizon, device=device).unsqueeze(0)
  ).reshape(-1)
  flat_obs = storage.observations.flatten(0, 1)
  flat_actions = storage.actions.flatten(0, 1)
  flat_log_prob = storage.actions_log_prob.flatten(0, 1)

  for _ in range(num_epochs):
    for batch_index in range(batches_per_epoch):
      start = batch_index * mini_batch_size
      stop = (
        data.num_samples
        if batch_index == batches_per_epoch - 1
        else (batch_index + 1) * mini_batch_size
      )
      aggregate_idx = trajectory_samples[start:stop]
      source_idx = data.source_indices[aggregate_idx]
      batch = RolloutStorage.Batch(
        observations=flat_obs[source_idx],
        actions=flat_actions[source_idx],
        values=data.values[aggregate_idx],
        advantages=data.advantages[aggregate_idx],
        returns=data.returns[aggregate_idx],
        old_actions_log_prob=flat_log_prob[source_idx],
        old_distribution_params=tuple(
          parameter[aggregate_idx] for parameter in data.kl_reference_params
        ),
      )
      batch.sapg_aggregate_indices = aggregate_idx
      batch.source_policy_ids = data.source_policy_ids[aggregate_idx]
      batch.target_policy_ids = data.target_policy_ids[aggregate_idx]
      batch.off_policy_mask = data.off_policy_mask[aggregate_idx]
      yield batch
