"""PPO optimization over official-compatible SAPG aggregate batches."""

from __future__ import annotations

import torch
import torch.nn as nn


def _update_learning_rate(algorithm, kl_mean: torch.Tensor) -> None:
  heft_scheduler = getattr(algorithm, "_update_heft_actor_lr", None)
  if callable(heft_scheduler):
    heft_scheduler(kl_mean)
    return

  has_split_lr = hasattr(algorithm, "actor_learning_rate") and hasattr(
    algorithm, "critic_learning_rate"
  )
  if algorithm.gpu_global_rank == 0:
    if kl_mean > algorithm.desired_kl * 2.0:
      scale = 1.0 / 1.5
    elif kl_mean < algorithm.desired_kl / 2.0 and kl_mean > 0.0:
      scale = 1.5
    else:
      scale = 1.0
    if has_split_lr:
      algorithm.actor_learning_rate = min(
        1.0e-2, max(1.0e-5, algorithm.actor_learning_rate * scale)
      )
      algorithm.critic_learning_rate = min(
        1.0e-2, max(1.0e-5, algorithm.critic_learning_rate * scale)
      )
    else:
      algorithm.learning_rate = min(
        1.0e-2, max(1.0e-5, algorithm.learning_rate * scale)
      )

  if algorithm.is_multi_gpu:
    if has_split_lr:
      rates = torch.tensor(
        [algorithm.actor_learning_rate, algorithm.critic_learning_rate],
        device=algorithm.device,
      )
      torch.distributed.broadcast(rates, src=0)
      algorithm.actor_learning_rate = float(rates[0].item())
      algorithm.critic_learning_rate = float(rates[1].item())
    else:
      rate = torch.tensor(algorithm.learning_rate, device=algorithm.device)
      torch.distributed.broadcast(rate, src=0)
      algorithm.learning_rate = float(rate.item())

  if has_split_lr:
    algorithm.learning_rate = algorithm.actor_learning_rate
    algorithm.optimizer.param_groups[0]["lr"] = algorithm.actor_learning_rate
    algorithm.optimizer.param_groups[1]["lr"] = algorithm.critic_learning_rate
  else:
    for group in algorithm.optimizer.param_groups:
      group["lr"] = algorithm.learning_rate


def _entropy_loss(algorithm, entropy: torch.Tensor, policy_ids: torch.Tensor):
  config = algorithm._sapg_runtime.config
  if config.exploration_type == "none":
    return algorithm.entropy_coef * entropy.mean()
  coefficients = torch.linspace(
    0.5,
    0.0,
    config.num_policy_blocks,
    device=entropy.device,
    dtype=entropy.dtype,
  ) * config.entropy_coef_scale
  return (coefficients[policy_ids] * entropy).mean()


def _clear_auxiliary_forward_caches(algorithm) -> None:
  # SPV6 normally reuses full-batch forward caches. SAPG filters duplicated
  # follower samples out of auxiliary objectives, so those caches no longer
  # share the same indexing and must be recomputed on the on-policy subset.
  for model in (algorithm.actor, algorithm.critic):
    for name in ("_cached_rma_latents", "_cached_normalized_history"):
      if hasattr(model, name):
        setattr(model, name, None)


def _on_policy_auxiliary_weight(
  on_policy_mask: torch.Tensor, batch_size: int
) -> tuple[int, float]:
  if batch_size <= 0:
    raise ValueError("SAPG auxiliary batch size must be positive")
  sample_count = int(on_policy_mask.sum().item())
  return sample_count, sample_count / batch_size


def update_sapg(algorithm) -> dict[str, float]:
  """Run SAPG without sharing any execution branch with disabled PPO."""
  runtime = algorithm._sapg_runtime
  context = runtime.context
  data = runtime.prepare_aggregated_data()

  mean_value_loss = 0.0
  mean_surrogate_loss = 0.0
  mean_entropy = 0.0
  mean_symmetry_loss = 0.0 if algorithm.symmetry else None
  mean_auxiliary_losses: dict[str, torch.Tensor] = {}
  auxiliary_samples = 0
  clipped_samples = 0
  off_policy_clipped_samples = 0
  off_policy_samples = 0
  num_updates = 0

  for batch in runtime.mini_batch_generator(data):
    num_updates += 1
    original_batch_size = batch.observations.batch_size[0]
    target_policy_ids = batch.target_policy_ids
    on_policy_mask = ~batch.off_policy_mask
    valid_mask = None
    get_valid_mask = getattr(algorithm, "_heft_valid_mask", None)
    if callable(get_valid_mask):
      valid_mask = get_valid_mask(batch.observations, original_batch_size)

    if algorithm.normalize_advantage_per_mini_batch:
      with torch.no_grad():
        batch.advantages = (batch.advantages - batch.advantages.mean()) / (
          batch.advantages.std() + 1.0e-8
        )

    if algorithm.symmetry:
      algorithm.symmetry.augment_batch(batch, original_batch_size)
    num_aug = batch.observations.batch_size[0] // original_batch_size
    augmented_policy_ids = target_policy_ids.repeat(num_aug)
    augmented_valid_mask = (
      valid_mask.repeat(num_aug, 1) if valid_mask is not None else None
    )

    with context.use(augmented_policy_ids):
      algorithm.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = algorithm.actor.get_output_log_prob(batch.actions)
    with context.use(augmented_policy_ids):
      values = algorithm.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )

    mirrored_values = None
    mirror_critic = getattr(algorithm, "_heft_mirrored_critic_values", None)
    if callable(mirror_critic):
      with context.use(target_policy_ids):
        mirrored_values = mirror_critic(
          batch.observations, original_batch_size
        )
    distribution_params = tuple(
      parameter[:original_batch_size]
      for parameter in algorithm.actor.output_distribution_params
    )
    entropy = algorithm.actor.output_entropy[:original_batch_size]

    if algorithm.desired_kl is not None and algorithm.schedule == "adaptive":
      with torch.inference_mode():
        kl = algorithm.actor.get_kl_divergence(
          batch.old_distribution_params, distribution_params
        )
        kl_mean = torch.mean(kl)
        if algorithm.is_multi_gpu:
          torch.distributed.all_reduce(
            kl_mean, op=torch.distributed.ReduceOp.SUM
          )
          kl_mean /= algorithm.gpu_world_size
        _update_learning_rate(algorithm, kl_mean)

    ratio = torch.exp(
      actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
    )
    surrogate = -torch.squeeze(batch.advantages) * ratio
    surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
      ratio, 1.0 - algorithm.clip_param, 1.0 + algorithm.clip_param
    )
    surrogate_terms = torch.max(surrogate, surrogate_clipped)
    if augmented_valid_mask is not None:
      surrogate_terms = surrogate_terms * augmented_valid_mask.squeeze(-1)
    surrogate_loss = surrogate_terms.mean()

    original_ratio = ratio[:original_batch_size]
    is_clipped = (original_ratio < 1.0 - algorithm.clip_param) | (
      original_ratio > 1.0 + algorithm.clip_param
    )
    clipped_samples += int(is_clipped.sum().item())
    off_policy_clipped_samples += int(
      (is_clipped & batch.off_policy_mask).sum().item()
    )
    off_policy_samples += int(batch.off_policy_mask.sum().item())

    if algorithm.use_clipped_value_loss:
      value_clipped = batch.values + (values - batch.values).clamp(
        -algorithm.clip_param, algorithm.clip_param
      )
      value_losses = (values - batch.returns).pow(2)
      value_losses_clipped = (value_clipped - batch.returns).pow(2)
      value_loss = torch.max(value_losses, value_losses_clipped).mean()
    else:
      value_errors = (batch.returns - values).pow(2)
      if augmented_valid_mask is not None:
        value_errors = value_errors * augmented_valid_mask
      value_loss = value_errors.mean()
    if mirrored_values is not None:
      mirror_targets = batch.returns[:original_batch_size]
      mirror_errors = (mirror_targets - mirrored_values).pow(2)
      if valid_mask is not None:
        mirror_errors = mirror_errors * valid_mask
      value_loss = 0.5 * (value_loss + mirror_errors.mean())

    entropy_loss = _entropy_loss(algorithm, entropy, target_policy_ids)
    loss = (
      surrogate_loss
      + algorithm.value_loss_coef * value_loss
      - entropy_loss
    )

    auxiliary_loss_fn = getattr(algorithm, "_auxiliary_loss", None)
    if callable(auxiliary_loss_fn):
      _clear_auxiliary_forward_caches(algorithm)
      on_policy_samples, auxiliary_weight = _on_policy_auxiliary_weight(
        on_policy_mask, original_batch_size
      )
      if on_policy_samples:
        auxiliary_obs = batch.observations[:original_batch_size][on_policy_mask]
        auxiliary_loss, auxiliary_metrics = auxiliary_loss_fn(auxiliary_obs)
        # Auxiliary objectives are defined on the original on-policy rollout.
        # Scale their subset mean back to a masked full-batch mean so adding
        # duplicated SAPG follower data does not increase their epoch weight.
        loss = loss + auxiliary_weight * auxiliary_loss
        auxiliary_samples += on_policy_samples
        for name, metric in auxiliary_metrics.items():
          detached = metric.detach() * on_policy_samples
          mean_auxiliary_losses[name] = (
            mean_auxiliary_losses[name] + detached
            if name in mean_auxiliary_losses
            else detached.clone()
          )

    get_std_symmetry_loss = getattr(algorithm.actor, "std_symmetry_loss", None)
    if callable(get_std_symmetry_loss):
      loss = loss + 10.0 * get_std_symmetry_loss()

    symmetry_loss = None
    if algorithm.symmetry:
      with context.use(target_policy_ids):
        symmetry_loss = algorithm.symmetry.compute_loss(
          algorithm.actor, batch, original_batch_size
        )
      if algorithm.symmetry.use_mirror_loss:
        loss = loss + algorithm.symmetry.mirror_loss_coeff * symmetry_loss

    algorithm.optimizer.zero_grad()
    zero_auxiliary_optimizers = getattr(
      algorithm, "_zero_auxiliary_optimizers", None
    )
    if callable(zero_auxiliary_optimizers):
      zero_auxiliary_optimizers()
    loss.backward()
    if algorithm.is_multi_gpu:
      algorithm.reduce_parameters()
    actor_parameters_for_clipping = getattr(
      algorithm, "_actor_parameters_for_gradient_clipping", None
    )
    actor_parameters = (
      actor_parameters_for_clipping()
      if callable(actor_parameters_for_clipping)
      else algorithm.actor.parameters()
    )
    nn.utils.clip_grad_norm_(actor_parameters, algorithm.max_grad_norm)
    nn.utils.clip_grad_norm_(algorithm.critic.parameters(), algorithm.max_grad_norm)
    clip_auxiliary_gradients = getattr(
      algorithm, "_clip_auxiliary_gradients", None
    )
    if callable(clip_auxiliary_gradients):
      clip_auxiliary_gradients()
    step_auxiliary_optimizers = getattr(
      algorithm, "_step_auxiliary_optimizers", None
    )
    if callable(step_auxiliary_optimizers):
      step_auxiliary_optimizers()
    algorithm.optimizer.step()

    mean_value_loss += value_loss.item()
    mean_surrogate_loss += surrogate_loss.item()
    mean_entropy += entropy.mean().item()
    if mean_symmetry_loss is not None and symmetry_loss is not None:
      mean_symmetry_loss += symmetry_loss.item()

  if num_updates == 0:
    raise RuntimeError("SAPG generated no optimization mini-batches")
  mean_value_loss /= num_updates
  mean_surrogate_loss /= num_updates
  mean_entropy /= num_updates
  if mean_symmetry_loss is not None:
    mean_symmetry_loss /= num_updates

  total_samples = data.num_samples
  loss_dict = {
    "value": mean_value_loss,
    "surrogate": mean_surrogate_loss,
    "entropy": mean_entropy,
    "sapg/off_policy_fraction": float(data.off_policy_mask.float().mean().item()),
    "sapg/clip_fraction": clipped_samples
    / max(1, total_samples * algorithm.num_learning_epochs),
    "sapg/off_policy_clip_fraction": off_policy_clipped_samples
    / max(1, off_policy_samples),
    "sapg/num_updates": float(num_updates),
  }
  for index, follower in enumerate(data.selected_follower_ids.tolist()):
    loss_dict[f"sapg/selected_follower_{index}"] = float(follower)
  distribution = algorithm.actor.distribution
  if distribution.std_type == "scalar":
    std_table = distribution.std_param.detach().clamp(
      distribution.std_range[0], distribution.std_range[1]
    )
  else:
    std_table = distribution.log_std_param.detach().clamp(
      distribution.log_std_range[0], distribution.log_std_range[1]
    ).exp()
  for policy_id, policy_std in enumerate(std_table):
    loss_dict[f"sapg/std_block_{policy_id}"] = float(policy_std.mean().item())
  if algorithm.symmetry:
    loss_dict["symmetry"] = mean_symmetry_loss
  if callable(getattr(algorithm.actor, "std_symmetry_loss", None)):
    loss_dict["symmetry_std"] = float(
      algorithm.actor.std_symmetry_loss().detach().item()
    )
  for name, total in mean_auxiliary_losses.items():
    loss_dict[name] = float((total / max(1, auxiliary_samples)).item())

  algorithm.storage.clear()
  runtime.clear()
  return loss_dict
