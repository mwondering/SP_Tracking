"""Tracking-specific PPO variants."""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer

from .optim import OptimizerGroup, build_heft_optimizer


class SparseTrackSplitLrPPO(PPO):
  """PPO with SparseTrack-style actor and critic optimizer learning rates."""

  _SPLIT_LR_STATE_KEY = "tracking_split_lr_state"

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    *args,
    actor_learning_rate: float | None = None,
    critic_learning_rate: float | None = None,
    clamp_rewards_min: float | None = None,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    **kwargs,
  ) -> None:
    super().__init__(
      actor,
      critic,
      storage,
      *args,
      learning_rate=learning_rate,
      optimizer=optimizer,
      **kwargs,
    )
    if actor_learning_rate is None or critic_learning_rate is None:
      self.clamp_rewards_min = clamp_rewards_min
      return

    self.actor_learning_rate = float(actor_learning_rate)
    self.critic_learning_rate = float(critic_learning_rate)
    self.clamp_rewards_min = clamp_rewards_min
    self.learning_rate = self.actor_learning_rate
    self.optimizer = resolve_optimizer(optimizer)(
      [
        {"params": self.actor.parameters(), "lr": self.actor_learning_rate},
        {"params": self.critic.parameters(), "lr": self.critic_learning_rate},
      ]
    )

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    """Optionally reproduce the reference task's non-negative reward signal."""
    if self.clamp_rewards_min is not None:
      rewards = rewards.clamp_min(float(self.clamp_rewards_min))
    return super().process_env_step(obs, rewards, dones, extras)

  def _set_learning_rate(self, lr: float) -> None:
    if hasattr(self, "actor_learning_rate") and hasattr(self, "critic_learning_rate"):
      old_actor_lr = self.actor_learning_rate
      old_critic_lr = self.critic_learning_rate
      self.actor_learning_rate = float(lr)
      self.critic_learning_rate = float(
        old_critic_lr * self.actor_learning_rate / old_actor_lr
      )
      self.learning_rate = self.actor_learning_rate
      self.optimizer.param_groups[0]["lr"] = self.actor_learning_rate
      self.optimizer.param_groups[1]["lr"] = self.critic_learning_rate
      return
    self.learning_rate = float(lr)
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = self.learning_rate

  def _split_lr_checkpoint_state(self) -> dict[str, float]:
    """Return mutable learning-rate state not covered by ``optimizer.state_dict``.

    The optimizer serializes its parameter-group learning rates, but this PPO
    variant also keeps Python-side actor/critic LR fields.  ``update`` uses
    those fields to overwrite the optimizer groups during adaptive scheduling,
    so they must round-trip with a resumed run as well.
    """
    if not (
      hasattr(self, "actor_learning_rate") and hasattr(self, "critic_learning_rate")
    ):
      return {}
    return {
      "actor_learning_rate": float(self.actor_learning_rate),
      "critic_learning_rate": float(self.critic_learning_rate),
      "learning_rate": float(self.learning_rate),
    }

  def _restore_split_lr_checkpoint_state(
    self, loaded_dict: dict, load_cfg: dict | None
  ) -> None:
    """Restore split LR fields after the base PPO restores its optimizer.

    Old checkpoints did not contain ``tracking_split_lr_state``.  For those,
    derive the authoritative values from the restored optimizer parameter
    groups so existing local runs remain resumable.
    """
    if not (
      hasattr(self, "actor_learning_rate") and hasattr(self, "critic_learning_rate")
    ):
      return
    if load_cfg is not None and not bool(load_cfg.get("optimizer", False)):
      return

    state = loaded_dict.get(self._SPLIT_LR_STATE_KEY, {})
    if not isinstance(state, dict):
      state = {}
    param_groups = self.optimizer.param_groups
    if len(param_groups) < 2:
      return

    actor_lr = state.get("actor_learning_rate", param_groups[0]["lr"])
    critic_lr = state.get("critic_learning_rate", param_groups[1]["lr"])
    self.actor_learning_rate = float(actor_lr)
    self.critic_learning_rate = float(critic_lr)
    self.learning_rate = float(state.get("learning_rate", self.actor_learning_rate))
    param_groups[0]["lr"] = self.actor_learning_rate
    param_groups[1]["lr"] = self.critic_learning_rate

  def save(self) -> dict:
    saved_dict = super().save()
    split_lr_state = self._split_lr_checkpoint_state()
    if split_lr_state:
      saved_dict[self._SPLIT_LR_STATE_KEY] = split_lr_state
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    self._restore_split_lr_checkpoint_state(loaded_dict, load_cfg)
    return load_iteration

  def update(self) -> dict[str, float]:
    """Run PPO update while preserving actor/critic split learning rates."""
    if not (
      hasattr(self, "actor_learning_rate") and hasattr(self, "critic_learning_rate")
    ):
      return super().update()

    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_rnd_loss = 0 if self.rnd else None
    mean_symmetry_loss = 0 if self.symmetry else None
    mean_auxiliary_losses: dict[str, torch.Tensor] = {}

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches,
        self.num_learning_epochs,
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches,
        self.num_learning_epochs,
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]
      valid_mask = None
      get_valid_mask = getattr(self, "_heft_valid_mask", None)
      if callable(get_valid_mask):
        valid_mask = get_valid_mask(batch.observations, original_batch_size)

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (  # type: ignore
            batch.advantages.std() + 1e-8
          )

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      mirrored_values = None
      mirror_critic = getattr(self, "_heft_mirrored_critic_values", None)
      if callable(mirror_critic):
        mirrored_values = mirror_critic(batch.observations, original_batch_size)
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(  # type: ignore
            batch.old_distribution_params,
            distribution_params,
          )
          kl_mean = torch.mean(kl)

          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size

          heft_scheduler = getattr(self, "_update_heft_actor_lr", None)
          if callable(heft_scheduler):
            heft_scheduler(kl_mean)
          else:
            if self.gpu_global_rank == 0:
              if kl_mean > self.desired_kl * 2.0:
                self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
              elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

            if self.is_multi_gpu:
              actor_lr_tensor = torch.tensor(self.actor_learning_rate, device=self.device)
              critic_lr_tensor = torch.tensor(
                self.critic_learning_rate,
                device=self.device,
              )
              torch.distributed.broadcast(actor_lr_tensor, src=0)
              torch.distributed.broadcast(critic_lr_tensor, src=0)
              self.actor_learning_rate = actor_lr_tensor.item()
              self.critic_learning_rate = critic_lr_tensor.item()

            self.learning_rate = self.actor_learning_rate
            self.optimizer.param_groups[0]["lr"] = self.actor_learning_rate
            self.optimizer.param_groups[1]["lr"] = self.critic_learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
      surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
        ratio,
        1.0 - self.clip_param,
        1.0 + self.clip_param,
      )
      surrogate_terms = torch.max(surrogate, surrogate_clipped)
      if valid_mask is not None:
        surrogate_terms = surrogate_terms * valid_mask.squeeze(-1)
      surrogate_loss = surrogate_terms.mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param,
          self.clip_param,
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_errors = (batch.returns - values).pow(2)
        if valid_mask is not None:
          value_errors = value_errors * valid_mask
        value_loss = value_errors.mean()
      if mirrored_values is not None:
        mirror_targets = batch.returns[:original_batch_size]
        mirror_errors = (mirror_targets - mirrored_values).pow(2)
        if valid_mask is not None:
          mirror_errors = mirror_errors * valid_mask
        value_loss = 0.5 * (
          value_loss + mirror_errors.mean()
        )

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )

      auxiliary_loss_fn = getattr(self, "_auxiliary_loss", None)
      if callable(auxiliary_loss_fn):
        auxiliary_loss, auxiliary_metrics = auxiliary_loss_fn(
          batch.observations[:original_batch_size]
        )
        loss = loss + auxiliary_loss
        for name, metric in auxiliary_metrics.items():
          detached = metric.detach()
          mean_auxiliary_losses[name] = (
            mean_auxiliary_losses[name] + detached
            if name in mean_auxiliary_losses
            else detached.clone()
          )

      std_symmetry_loss = None
      get_std_symmetry_loss = getattr(self.actor, "std_symmetry_loss", None)
      if callable(get_std_symmetry_loss):
        std_symmetry_loss = get_std_symmetry_loss()
        loss = loss + 10.0 * std_symmetry_loss

      rnd_loss = (
        self.rnd.compute_loss(batch.observations[:original_batch_size])
        if self.rnd
        else None
      )

      if self.symmetry:
        symmetry_loss = self.symmetry.compute_loss(
          self.actor,
          batch,
          original_batch_size,
        )
        if self.symmetry.use_mirror_loss:
          loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

      self.optimizer.zero_grad()
      zero_auxiliary_optimizers = getattr(
        self, "_zero_auxiliary_optimizers", None
      )
      if callable(zero_auxiliary_optimizers):
        zero_auxiliary_optimizers()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        assert rnd_loss is not None
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      step_auxiliary_optimizers = getattr(
        self, "_step_auxiliary_optimizers", None
      )
      if callable(step_auxiliary_optimizers):
        step_auxiliary_optimizers()
      self.optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      if mean_rnd_loss is not None:
        assert rnd_loss is not None
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    if callable(getattr(self.actor, "std_symmetry_loss", None)):
      loss_dict["symmetry_std"] = float(
        self.actor.std_symmetry_loss().detach().item()
      )
    for name, total in mean_auxiliary_losses.items():
      loss_dict[name] = float((total / num_updates).item())

    self.storage.clear()
    return loss_dict

  def reduce_parameters(self) -> None:
    """Collect gradients from all GPUs and average them."""
    all_params = chain(self.actor.parameters(), self.critic.parameters())
    if self.rnd:
      all_params = chain(all_params, self.rnd.parameters())
    all_params = list(all_params)
    grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
    all_grads = torch.cat(grads)
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size
    offset = 0
    for param in all_params:
      if param.grad is not None:
        numel = param.numel()
        param.grad.data.copy_(
          all_grads[offset : offset + numel].view_as(param.grad.data)
        )
        offset += numel


class SPV3EstimatorPPO(SparseTrackSplitLrPPO):
  """PPO plus explicit SPV3 root-state estimator supervision."""

  _ESTIMATOR_OPTIMIZER_STATE_KEY = "estimator_optimizer_state_dict"

  def __init__(
    self,
    *args,
    estimator_learning_rate: float = 1.0e-4,
    estimator_root_height_loss_coef: float = 1.0,
    estimator_root_lin_vel_loss_coef: float = 1.0,
    **kwargs,
  ) -> None:
    optimizer_name = str(kwargs.get("optimizer", "adam"))
    super().__init__(*args, **kwargs)
    if not (
      hasattr(self, "actor_learning_rate")
      and hasattr(self, "critic_learning_rate")
    ):
      raise ValueError(
        "SPV3EstimatorPPO requires actor_learning_rate and "
        "critic_learning_rate so the estimator can be stepped independently"
      )
    estimator = getattr(self.actor, "estimator", None)
    if not isinstance(estimator, nn.Module):
      raise TypeError("SPV3EstimatorPPO requires actor.estimator to be a module")
    self.estimator_learning_rate = float(estimator_learning_rate)
    if self.estimator_learning_rate <= 0.0:
      raise ValueError("estimator_learning_rate must be positive")
    self._estimator_parameters = tuple(estimator.parameters())
    if not self._estimator_parameters:
      raise ValueError("actor.estimator has no trainable parameters")
    self.estimator_optimizer = resolve_optimizer(optimizer_name)(
      self._estimator_parameters,
      lr=self.estimator_learning_rate,
    )
    self.estimator_root_height_loss_coef = float(
      estimator_root_height_loss_coef
    )
    self.estimator_root_lin_vel_loss_coef = float(
      estimator_root_lin_vel_loss_coef
    )

  def _auxiliary_loss(
    self, observations
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    estimator_losses = getattr(self.actor, "estimator_losses", None)
    if not callable(estimator_losses):
      raise TypeError("SPV3EstimatorPPO requires an actor with estimator_losses()")
    height_mse, lin_vel_mse = estimator_losses(observations)
    loss = (
      self.estimator_root_height_loss_coef * height_mse
      + self.estimator_root_lin_vel_loss_coef * lin_vel_mse
    )
    return loss, {
      "estimator_root_height_mse": height_mse,
      "estimator_root_lin_vel_mse": lin_vel_mse,
    }

  def _step_auxiliary_optimizers(self) -> None:
    """Step the estimator once and keep it out of the actor optimizer step.

    The main optimizer deliberately retains its legacy two-group parameter
    layout so checkpoints produced before the independent estimator optimizer
    remain loadable.  Clearing estimator gradients after this step makes the
    following main optimizer step skip those parameters.
    """
    self.estimator_optimizer.step()
    for parameter in self._estimator_parameters:
      parameter.grad = None

  def _zero_auxiliary_optimizers(self) -> None:
    self.estimator_optimizer.zero_grad()

  def update(self) -> dict[str, float]:
    result = super().update()
    result["estimator_lr"] = float(
      self.estimator_optimizer.param_groups[0]["lr"]
    )
    return result

  def save(self) -> dict:
    saved_dict = super().save()
    saved_dict[self._ESTIMATOR_OPTIMIZER_STATE_KEY] = (
      self.estimator_optimizer.state_dict()
    )
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    load_optimizer = load_cfg is None or bool(load_cfg.get("optimizer", False))
    estimator_state = loaded_dict.get(self._ESTIMATOR_OPTIMIZER_STATE_KEY)
    if load_optimizer and isinstance(estimator_state, dict):
      self.estimator_optimizer.load_state_dict(estimator_state)
      self.estimator_learning_rate = float(
        self.estimator_optimizer.param_groups[0]["lr"]
      )
    return load_iteration


class SPV5ReferenceEncoderPPO(SPV3EstimatorPPO):
  """SPV3 supervision plus the normalized equal-MSE reference objective."""

  @staticmethod
  def construct_algorithm(obs, env, cfg: dict, device: str):
    # Reserve the compact behavior-time cache in rollout storage before RSL
    # allocates its TensorDict buffers.  It contains only derived reference
    # state (577) and the detached root estimate (4), not supervision targets.
    from .spv5_models import (
      SPV5_POLICY_CONTEXT_CACHE_DIM,
      SPV5_POLICY_CONTEXT_CACHE_GROUP,
    )

    cache_device = obs.device
    if cache_device is None:
      cache_device = next(iter(obs.values())).device
    obs.set(
      SPV5_POLICY_CONTEXT_CACHE_GROUP,
      torch.zeros(
        (*obs.batch_size, SPV5_POLICY_CONTEXT_CACHE_DIM),
        device=cache_device,
      ),
    )
    algorithm = PPO.construct_algorithm(obs, env, cfg, device)
    populate = getattr(algorithm.actor, "populate_policy_context_cache", None)
    if not callable(populate):
      raise TypeError(
        "SPV5ReferenceEncoderPPO requires an actor with "
        "populate_policy_context_cache()"
      )
    populate(obs)
    return algorithm

  def __init__(
    self,
    *args,
    reference_encoder_loss_coef: float = 1.0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.reference_encoder_loss_coef = float(reference_encoder_loss_coef)

  def _auxiliary_loss(
    self, observations
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    estimator_loss, diagnostics = super()._auxiliary_loss(observations)
    reference_losses = getattr(self.actor, "reference_encoder_losses", None)
    if not callable(reference_losses):
      raise TypeError(
        "SPV5ReferenceEncoderPPO requires an actor with "
        "reference_encoder_losses()"
      )
    reference_loss, reference_diagnostics = reference_losses(observations)
    diagnostics.update(reference_diagnostics)
    return (
      estimator_loss + self.reference_encoder_loss_coef * reference_loss,
      diagnostics,
    )


class SPV51ContactEstimatorPPO(SPV5ReferenceEncoderPPO):
  """SPV5 objectives plus binary left/right foot-contact supervision."""

  @staticmethod
  def construct_algorithm(obs, env, cfg: dict, device: str):
    from .spv5_1_models import (
      SPV5_1_POLICY_CONTEXT_CACHE_DIM,
      SPV5_1_POLICY_CONTEXT_CACHE_GROUP,
    )

    cache_device = obs.device
    if cache_device is None:
      cache_device = next(iter(obs.values())).device
    obs.set(
      SPV5_1_POLICY_CONTEXT_CACHE_GROUP,
      torch.zeros(
        (*obs.batch_size, SPV5_1_POLICY_CONTEXT_CACHE_DIM),
        device=cache_device,
      ),
    )
    algorithm = PPO.construct_algorithm(obs, env, cfg, device)
    populate = getattr(algorithm.actor, "populate_policy_context_cache", None)
    if not callable(populate):
      raise TypeError(
        "SPV51ContactEstimatorPPO requires an actor with "
        "populate_policy_context_cache()"
      )
    populate(obs)
    return algorithm

  def __init__(
    self,
    *args,
    estimator_foot_contact_loss_coef: float = 0.1,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.estimator_foot_contact_loss_coef = float(
      estimator_foot_contact_loss_coef
    )

  def _auxiliary_loss(
    self, observations
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    estimator_losses = getattr(
      self.actor, "estimator_contact_losses", None
    )
    if not callable(estimator_losses):
      raise TypeError(
        "SPV51ContactEstimatorPPO requires an actor with "
        "estimator_contact_losses()"
      )
    height_mse, lin_vel_mse, contact_bce, contact_diagnostics = (
      estimator_losses(observations)
    )
    estimator_loss = (
      self.estimator_root_height_loss_coef * height_mse
      + self.estimator_root_lin_vel_loss_coef * lin_vel_mse
      + self.estimator_foot_contact_loss_coef * contact_bce
    )

    reference_losses = getattr(self.actor, "reference_encoder_losses", None)
    if not callable(reference_losses):
      raise TypeError(
        "SPV51ContactEstimatorPPO requires an actor with "
        "reference_encoder_losses()"
      )
    reference_loss, reference_diagnostics = reference_losses(observations)
    diagnostics = {
      "estimator_root_height_mse": height_mse,
      "estimator_root_lin_vel_mse": lin_vel_mse,
      "estimator_foot_contact_bce": contact_bce,
      **contact_diagnostics,
      **reference_diagnostics,
    }
    return (
      estimator_loss + self.reference_encoder_loss_coef * reference_loss,
      diagnostics,
    )


class SPV6RmaPPO(SPV5ReferenceEncoderPPO):
  """SPV5 objectives plus asymmetric RMA alignment and reconstruction."""

  def __init__(
    self,
    *args,
    rma_global_alignment_coef: float = 1.0,
    rma_sensor_alignment_coef: float = 0.5,
    rma_push_alignment_coef: float = 1.0,
    rma_physics_reconstruction_coef: float = 0.1,
    rma_push_reconstruction_coef: float = 0.1,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.rma_global_alignment_coef = float(rma_global_alignment_coef)
    self.rma_sensor_alignment_coef = float(rma_sensor_alignment_coef)
    self.rma_push_alignment_coef = float(rma_push_alignment_coef)
    self.rma_physics_reconstruction_coef = float(
      rma_physics_reconstruction_coef
    )
    self.rma_push_reconstruction_coef = float(rma_push_reconstruction_coef)

  def _auxiliary_loss(self, observations):
    base_loss, diagnostics = super()._auxiliary_loss(observations)
    actor_latents = getattr(self.actor, "rma_latents", None)
    critic_latents = getattr(self.critic, "rma_latents", None)
    reconstruction_losses = getattr(self.critic, "reconstruction_losses", None)
    if not callable(actor_latents) or not callable(critic_latents):
      raise TypeError("SPV6RmaPPO requires actor and critic RMA latent methods")
    if not callable(reconstruction_losses):
      raise TypeError("SPV6RmaPPO requires critic reconstruction_losses()")
    batch_size = observations.batch_size[0]
    actor_cached = getattr(self.actor, "cached_rma_latents", None)
    critic_cached = getattr(self.critic, "cached_rma_latents", None)
    cached_actor_values = (
      actor_cached(batch_size) if callable(actor_cached) else None
    )
    cached_critic_values = (
      critic_cached(batch_size) if callable(critic_cached) else None
    )
    actor_global, actor_sensor, actor_push = (
      actor_latents(observations)
      if cached_actor_values is None
      else cached_actor_values
    )
    critic_global, critic_sensor, critic_push = (
      critic_latents(observations)
      if cached_critic_values is None
      else cached_critic_values
    )
    global_alignment = (
      actor_global - critic_global.detach()
    ).square().mean()
    sensor_alignment = (
      actor_sensor - critic_sensor.detach()
    ).square().mean()
    push_alignment = (actor_push - critic_push.detach()).square().mean()
    physics_reconstruction, push_reconstruction, recon_diagnostics = (
      reconstruction_losses(
        observations, (critic_global, critic_sensor, critic_push)
      )
    )
    rma_loss = (
      self.rma_global_alignment_coef * global_alignment
      + self.rma_sensor_alignment_coef * sensor_alignment
      + self.rma_push_alignment_coef * push_alignment
      + self.rma_physics_reconstruction_coef * physics_reconstruction
      + self.rma_push_reconstruction_coef * push_reconstruction
    )
    diagnostics.update(recon_diagnostics)
    diagnostics.update(
      {
        "rma_alignment_global": global_alignment,
        "rma_alignment_sensor": sensor_alignment,
        "rma_alignment_push": push_alignment,
        "rma_latent_global_std": critic_global.std(),
        "rma_latent_sensor_std": critic_sensor.std(),
        "rma_latent_push_std": critic_push.std(),
      }
    )
    return base_loss + rma_loss, diagnostics


def _schedule_value(schedule, progress: float) -> float:
  if isinstance(schedule, (int, float)):
    return float(schedule)
  points = [(float(x), float(y)) for x, y in schedule]
  if progress <= points[0][0]:
    return points[0][1]
  for (x0, y0), (x1, y1) in zip(points, points[1:]):
    if progress <= x1:
      return y0 + (y1 - y0) * (progress - x0) / (x1 - x0)
  return points[-1][1]


class HeftTeacherPPO(SparseTrackSplitLrPPO):
  """Teacher-only HEFT pretrain optimizer and schedules on RSL storage."""

  _HEFT_STATE_KEY = "heft_teacher_pretrain_state"

  @staticmethod
  def construct_algorithm(obs, env, cfg: dict, device: str):
    obs["is_init"] = torch.ones(
      (*obs.batch_size, 1), dtype=torch.bool, device=obs.device
    )
    algorithm = PPO.construct_algorithm(obs, env, cfg, device)
    algorithm._heft_next_is_init = torch.ones(
      (env.num_envs, 1), dtype=torch.bool, device=device
    )
    return algorithm

  def __init__(
    self,
    actor,
    critic,
    storage,
    *args,
    actor_learning_rate: float = 1.0e-4,
    critic_learning_rate: float = 5.0e-4,
    entropy_coef_start: float = 0.01,
    entropy_coef_end: float = 0.005,
    desired_kl_upper=((0.0, 0.015), (0.15, 0.015), (0.2, 0.01), (0.8, 0.0075), (1.0, 0.0075)),
    lr_schedule_scale_factor: float = 1.05,
    lr_schedule_min: float = 1.0e-7,
    lr_schedule_max: float = 1.0e-3,
    optimizer: str = "muon",
    **kwargs,
  ) -> None:
    if optimizer != "muon":
      raise ValueError("HeftTeacherPPO requires optimizer=muon")
    kwargs.pop("entropy_coef", None)
    super().__init__(
      actor,
      critic,
      storage,
      *args,
      actor_learning_rate=actor_learning_rate,
      critic_learning_rate=critic_learning_rate,
      entropy_coef=entropy_coef_start,
      optimizer="adam",
      **kwargs,
    )
    actor_optimizer = build_heft_optimizer(
      actor.parameters(),
      lr=actor_learning_rate,
      adamw_only=actor.adamw_only_parameters(),
    )
    critic_optimizer = build_heft_optimizer(
      critic.parameters(),
      lr=critic_learning_rate,
      adamw_only=critic.adamw_only_parameters(),
    )
    self.optimizer = OptimizerGroup([actor_optimizer, critic_optimizer])
    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer
    self.entropy_coef_start = float(entropy_coef_start)
    self.entropy_coef_end = float(entropy_coef_end)
    self.desired_kl_upper = desired_kl_upper
    self.lr_schedule_scale_factor = float(lr_schedule_scale_factor)
    self.lr_schedule_min = float(lr_schedule_min)
    self.lr_schedule_max = float(lr_schedule_max)
    self.progress = 0.0

  @staticmethod
  def _set_optimizer_lr(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
      group["lr"] = float(value)

  def step_schedule(self, progress: float, iteration: int) -> dict[str, float]:
    del iteration
    self.progress = max(0.0, min(float(progress), 1.0))
    self.entropy_coef = self.entropy_coef_start * (
      self.entropy_coef_end / self.entropy_coef_start
    ) ** self.progress
    return {"entropy_coef": self.entropy_coef, "progress": self.progress}

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    """Match HEFT's symmetric VecNorm update before storing a transition."""
    if self.symmetry is not None:
      normalized_obs, _ = self.symmetry.data_augmentation_func(
        env=self.symmetry.env, obs=obs, actions=None
      )
    else:
      normalized_obs = obs
    self.actor.update_normalization(normalized_obs)
    self.critic.update_normalization(normalized_obs)
    if self.clamp_rewards_min is not None:
      rewards = rewards.clamp_min(float(self.clamp_rewards_min))
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
        1,
      )
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self._heft_next_is_init = dones.reshape(-1, 1).bool()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def act(self, obs):
    if not hasattr(self, "_heft_next_is_init"):
      self._heft_next_is_init = torch.ones(
        (obs.batch_size[0], 1), dtype=torch.bool, device=obs.device
      )
    obs["is_init"] = self._heft_next_is_init
    normalizer = getattr(self.actor, "obs_normalizer", None)
    count = getattr(normalizer, "count", None)
    if isinstance(count, torch.Tensor) and not bool((count > 0).all()):
      if self.symmetry is not None:
        normalization_obs, _ = self.symmetry.data_augmentation_func(
          env=self.symmetry.env, obs=obs, actions=None
        )
      else:
        normalization_obs = obs
      self.actor.update_normalization(normalization_obs)
      self.critic.update_normalization(normalization_obs)
    return super().act(obs)

  def compute_returns(self, obs) -> None:
    """Compute GAE with HEFT's global (cross-rank) advantage statistics."""
    st = self.storage
    critic_hidden_state = self.critic.get_hidden_state()
    last_values = self.critic(obs).detach()
    self.critic.reset(hidden_state=critic_hidden_state)
    advantage = 0
    for step in reversed(range(st.num_transitions_per_env)):
      next_values = (
        last_values
        if step == st.num_transitions_per_env - 1
        else st.values[step + 1]
      )
      alive = 1.0 - st.dones[step].float()
      delta = st.rewards[step] + alive * self.gamma * next_values - st.values[step]
      advantage = delta + alive * self.gamma * self.lam * advantage
      st.returns[step] = advantage + st.values[step]
    st.advantages = st.returns - st.values
    if self.normalize_advantage_per_mini_batch:
      return
    valid = (~st.observations["is_init"]).to(st.advantages.dtype)
    total = (st.advantages * valid).sum()
    square_total = (st.advantages.square() * valid).sum()
    count = valid.sum().clamp_min(1.0)
    if self.is_multi_gpu:
      packed = torch.stack((total, square_total, count))
      torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
      total, square_total, count = packed
    mean = total / count
    variance = (square_total / count - mean.square()).clamp_min(0.0)
    st.advantages = (st.advantages - mean) / (variance.sqrt() + 1.0e-8)

  @staticmethod
  def _heft_valid_mask(observations, original_batch_size: int):
    return (~observations["is_init"][:original_batch_size]).to(torch.float32)

  def _heft_mirrored_critic_values(self, observations, original_batch_size: int):
    if self.symmetry is None:
      return None
    augmented, _ = self.symmetry.data_augmentation_func(
      env=self.symmetry.env,
      obs=observations[:original_batch_size],
      actions=None,
    )
    if augmented is None:
      return None
    return self.critic(augmented[original_batch_size:])

  def _update_heft_actor_lr(self, kl: torch.Tensor) -> None:
    upper = _schedule_value(self.desired_kl_upper, self.progress)
    kl_value = float(kl.item())
    new_lr = self.actor_learning_rate
    if kl_value > upper:
      new_lr = max(self.lr_schedule_min, new_lr / self.lr_schedule_scale_factor)
    if self.is_multi_gpu:
      value = torch.tensor(new_lr, device=self.device)
      torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
      new_lr = float((value / self.gpu_world_size).item())
    self.actor_learning_rate = new_lr
    self.learning_rate = new_lr
    self._set_optimizer_lr(self._actor_optimizer, new_lr)

  def update(self) -> dict[str, float]:
    result = super().update()
    self.actor.clamp_std()
    result["entropy_coef"] = float(self.entropy_coef)
    result["actor_lr"] = float(self.actor_learning_rate)
    result["critic_lr"] = float(self.critic_learning_rate)
    return result

  def save(self) -> dict:
    state = super().save()
    state[self._HEFT_STATE_KEY] = {
      "progress": self.progress,
      "entropy_coef": self.entropy_coef,
      "actor_learning_rate": self.actor_learning_rate,
      "critic_learning_rate": self.critic_learning_rate,
    }
    return state

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    state = loaded_dict.get(self._HEFT_STATE_KEY, {})
    self.progress = float(state.get("progress", self.progress))
    self.entropy_coef = float(state.get("entropy_coef", self.entropy_coef))
    self.actor_learning_rate = float(
      state.get("actor_learning_rate", self.actor_learning_rate)
    )
    self.critic_learning_rate = float(
      state.get("critic_learning_rate", self.critic_learning_rate)
    )
    self._set_optimizer_lr(self._actor_optimizer, self.actor_learning_rate)
    self._set_optimizer_lr(self._critic_optimizer, self.critic_learning_rate)
    return load_iteration
