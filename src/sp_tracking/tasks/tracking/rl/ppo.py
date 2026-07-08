"""Tracking-specific PPO variants."""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class SparseTrackSplitLrPPO(PPO):
  """PPO with SparseTrack-style actor and critic optimizer learning rates."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    *args,
    actor_learning_rate: float | None = None,
    critic_learning_rate: float | None = None,
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
      return

    self.actor_learning_rate = float(actor_learning_rate)
    self.critic_learning_rate = float(critic_learning_rate)
    self.learning_rate = self.actor_learning_rate
    self.optimizer = resolve_optimizer(optimizer)(
      [
        {"params": self.actor.parameters(), "lr": self.actor_learning_rate},
        {"params": self.critic.parameters(), "lr": self.critic_learning_rate},
      ]
    )

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
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param,
          self.clip_param,
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )

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
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        assert rnd_loss is not None
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
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
