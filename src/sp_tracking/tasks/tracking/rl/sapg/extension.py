"""SAPG/CPO runtime lifecycle and algorithm construction."""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from .batch import (
  SAPGAggregatedData,
  build_aggregated_data,
  build_cpo_aggregated_data,
  normalize_aggregated_advantages,
  rollout_policy_ids,
  sapg_mini_batch_generator,
)
from .conditioning import PolicyContext, install_policy_conditioning
from .config import SAPGConfig


class SAPGRuntime:
  """State owned only by algorithms constructed with SAPG or CPO enabled."""

  _SCHEMA_VERSION = 1

  def __init__(
    self,
    algorithm,
    config: SAPGConfig,
    context: PolicyContext,
  ) -> None:
    self.algorithm = algorithm
    self.config = config
    self.context = context
    self.last_observations: TensorDict | None = None
    self.last_selected_follower_ids = torch.empty(0, dtype=torch.long)
    self._rng = torch.Generator(device="cpu")
    self._rng.manual_seed(torch.initial_seed())

  def rollout_ids(
    self, num_envs: int, device: torch.device | str, *, training: bool
  ) -> torch.Tensor:
    return rollout_policy_ids(
      num_envs,
      self.config.num_policy_blocks,
      device,
      require_divisible=training,
    )

  def capture_last_observations(self, observations: TensorDict) -> None:
    self.last_observations = observations.detach().clone()

  def _select_followers(self) -> torch.Tensor:
    count = self.config.off_policy_ratio
    if self.algorithm.gpu_global_rank == 0:
      selected = torch.randperm(
        self.config.num_policy_blocks - 1,
        generator=self._rng,
      )[:count].to(self.algorithm.device)
    else:
      selected = torch.empty(count, dtype=torch.long, device=self.algorithm.device)
    if self.algorithm.is_multi_gpu:
      torch.distributed.broadcast(selected, src=0)
    self.last_selected_follower_ids = selected.detach().cpu()
    return selected

  def _critic_values(
    self,
    observations: TensorDict,
    policy_ids: torch.Tensor,
    indices: torch.Tensor | None = None,
  ) -> torch.Tensor:
    values: list[torch.Tensor] = []
    chunk_size = self.config.value_eval_chunk_size
    batch_size = (
      int(indices.numel()) if indices is not None else observations.batch_size[0]
    )
    with torch.no_grad():
      for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        chunk = (
          observations[indices[start:stop]]
          if indices is not None
          else observations[start:stop]
        )
        with self.context.use(policy_ids[start:stop]):
          values.append(self.algorithm.critic(chunk).detach())
    return torch.cat(values)

  def prepare_aggregated_data(self) -> SAPGAggregatedData:
    storage = self.algorithm.storage
    num_envs = int(storage.num_envs)
    horizon = int(storage.num_transitions_per_env)
    rollout_policy_ids(
      num_envs,
      self.config.num_policy_blocks,
      storage.values.device,
      require_divisible=True,
    )
    if self.last_observations is None:
      raise RuntimeError(
        f"{self.config.method.upper()} compute_returns() must run before update()"
      )

    selected = self._select_followers()
    block_size = num_envs // self.config.num_policy_blocks
    off_envs = torch.cat(
      [
        torch.arange(
          int(follower) * block_size,
          (int(follower) + 1) * block_size,
          device=storage.values.device,
        )
        for follower in selected
      ]
    )
    current_indices = (
      torch.arange(horizon, device=storage.values.device).unsqueeze(1) * num_envs
      + off_envs.unsqueeze(0)
    ).reshape(-1)
    leader_ids = torch.full(
      (current_indices.numel(),),
      self.config.leader_policy_id,
      dtype=torch.long,
      device=storage.values.device,
    )
    leader_values = self._critic_values(
      storage.observations.flatten(0, 1),
      leader_ids,
      current_indices,
    ).reshape(horizon, off_envs.numel(), 1)
    last_obs = self.last_observations[off_envs]
    last_ids = torch.full(
      (off_envs.numel(),),
      self.config.leader_policy_id,
      dtype=torch.long,
      device=storage.values.device,
    )
    last_values = self._critic_values(last_obs, last_ids)
    leader_next_values = torch.cat((leader_values[1:], last_values.unsqueeze(0)))

    if self.config.is_cpo:
      leader_envs = torch.arange(
        self.config.leader_policy_id * block_size,
        num_envs,
        device=storage.values.device,
      )
      num_followers = self.config.num_policy_blocks - 1
      leader_copy_envs = leader_envs.repeat(num_followers)
      follower_ids = torch.arange(
        num_followers,
        dtype=torch.long,
        device=storage.values.device,
      ).repeat_interleave(block_size)
      copy_indices = (
        torch.arange(horizon, device=storage.values.device).unsqueeze(1)
        * num_envs
        + leader_copy_envs.unsqueeze(0)
      ).reshape(-1)
      repeated_follower_ids = follower_ids.repeat(horizon)
      follower_values = self._critic_values(
        storage.observations.flatten(0, 1),
        repeated_follower_ids,
        copy_indices,
      ).reshape(horizon, leader_copy_envs.numel(), 1)
      follower_last_values = self._critic_values(
        self.last_observations[leader_copy_envs],
        follower_ids,
      )
      follower_next_values = torch.cat(
        (follower_values[1:], follower_last_values.unsqueeze(0))
      )
      data = build_cpo_aggregated_data(
        storage,
        selected,
        leader_values,
        leader_next_values,
        follower_values,
        follower_next_values,
        num_policy_blocks=self.config.num_policy_blocks,
        gamma=self.algorithm.gamma,
      )
    else:
      data = build_aggregated_data(
        storage,
        selected,
        leader_values,
        leader_next_values,
        num_policy_blocks=self.config.num_policy_blocks,
        gamma=self.algorithm.gamma,
      )
    if not self.algorithm.normalize_advantage_per_mini_batch:
      normalize_aggregated_advantages(
        data,
        storage,
        is_multi_gpu=self.algorithm.is_multi_gpu,
      )
    return data

  def mini_batch_generator(self, data: SAPGAggregatedData):
    return sapg_mini_batch_generator(
      self.algorithm.storage,
      data,
      num_mini_batches=self.algorithm.num_mini_batches,
      num_epochs=self.algorithm.num_learning_epochs,
    )

  def clear(self) -> None:
    self.last_observations = None

  def save(self) -> dict[str, Any]:
    return {
      "schema_version": self._SCHEMA_VERSION,
      "config": self.config.to_dict(),
      "rng_state": self._rng.get_state(),
    }

  def load(self, state: dict[str, Any]) -> None:
    if not state:
      return
    version = int(state.get("schema_version", 0))
    if version != self._SCHEMA_VERSION:
      raise ValueError(
        f"Unsupported ensemble checkpoint schema version {version}; "
        f"expected {self._SCHEMA_VERSION}"
      )
    saved_config = SAPGConfig.from_dict(state.get("config"))
    if saved_config != self.config:
      raise ValueError(
        "Ensemble checkpoint configuration does not match the current run: "
        f"saved={saved_config.to_dict()}, current={self.config.to_dict()}"
      )
    rng_state = state.get("rng_state")
    if isinstance(rng_state, torch.Tensor):
      self._rng.set_state(rng_state.cpu())


def construct_sapg_algorithm(obs, env, cfg: dict, device: str):
  """Inject SAPG/CPO policy conditioning before optimizer creation."""
  sapg_config = SAPGConfig.from_dict(cfg["algorithm"].get("sapg_cfg"))
  if not sapg_config.enabled:
    raise ValueError(
      "construct_sapg_algorithm requires an enabled SAPG/CPO configuration"
    )
  if cfg.get("torch_compile_mode") is not None:
    raise ValueError(
      f"{sapg_config.method.upper()} does not support torch_compile_mode"
    )
  if cfg["algorithm"].get("rnd_cfg") is not None:
    raise ValueError(
      f"{sapg_config.method.upper()} official compatibility does not support RND"
    )

  alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
  actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
  critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

  default_sets = ["actor", "critic"]
  cfg["obs_groups"] = resolve_obs_groups(
    obs, cfg["obs_groups"], default_sets
  )
  cfg["algorithm"] = resolve_rnd_config(
    cfg["algorithm"], obs, cfg["obs_groups"], env
  )
  cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

  actor = actor_class(
    obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
  ).to(device)
  print(f"Actor Model: {actor}")
  if cfg["algorithm"].pop("share_cnn_encoders", None):
    cfg["critic"]["cnns"] = actor.cnns
  critic = critic_class(
    obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
  ).to(device)
  print(f"Critic Model: {critic}")

  install_policy_conditioning(actor, critic, sapg_config)
  storage = RolloutStorage(
    "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
  )
  algorithm = alg_class(
    actor,
    critic,
    storage,
    device=device,
    **cfg["algorithm"],
    multi_gpu_cfg=cfg["multi_gpu"],
  )
  algorithm.compile(None)
  return algorithm
