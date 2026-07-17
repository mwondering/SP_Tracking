"""SPV6-1 oracle models with direct DR and push-history observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from sp_tracking.tasks.tracking.mdp.spv4 import SPV4_KEY_BODY_STATE_DIM
from sp_tracking.tasks.tracking.mdp.spv5 import (
  SPV5_REFERENCE_INPUT_DIM,
  SPV5_REFERENCE_TARGET_DIM,
  SPV5_ROBOT_ROOT_QUAT_DIM,
)
from sp_tracking.tasks.tracking.mdp.spv6 import (
  SPV6_PHYSICS_DIM,
  SPV6_PUSH_HISTORY_DIM,
)

from .heft_models import DecayVecNorm, _make_mlp, _orthogonal_small_
from .spv3_models import PROPRIO_TERM_DIMS, _identity_or_normalizer
from .spv5_models import (
  SPV5_POLICY_INPUT_DIM,
  SPV5_REFERENCE_CACHE_DIM,
  SPV5ReferenceEncoderActor,
  _SPV5ActorExport,
  _normalizer_inverse,
  _spv5_policy_features,
)


SPV6_1_DIRECT_DIM = SPV6_PHYSICS_DIM + SPV6_PUSH_HISTORY_DIM


class SPV61DirectActor(SPV5ReferenceEncoderActor):
  """SPV5 actor receiving actual DR parameters and push history directly."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    *,
    rma_physics_actual_group: str = "rma_physics_actual",
    rma_push_history_group: str = "rma_push_history",
    **kwargs,
  ) -> None:
    self.physics_group = str(rma_physics_actual_group)
    self.push_group = str(rma_push_history_group)
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      extra_actor_groups=(self.physics_group, self.push_group),
      extra_policy_input_dim=SPV6_1_DIRECT_DIM,
      raw_actor_obs_extra_dim=SPV6_1_DIRECT_DIM,
      **kwargs,
    )
    if int(obs[self.physics_group].shape[-1]) != SPV6_PHYSICS_DIM:
      raise ValueError("SPV6-1 physical parameter dimension mismatch")
    if int(obs[self.push_group].shape[-1]) != SPV6_PUSH_HISTORY_DIM:
      raise ValueError("SPV6-1 push history dimension mismatch")
    # Only the inherited SPV5 policy features are normalized.  Actual physics
    # and the recorded push window enter the policy exactly as provided.
    self.policy_normalizer = _identity_or_normalizer(
      self.obs_normalization, SPV5_POLICY_INPUT_DIM
    )

  def _direct_observations(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat((obs[self.physics_group], obs[self.push_group]), dim=-1)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    context = obs.get("spv5_policy_context_cache")
    if context is not None:
      features = self._spv5_features(
        obs,
        context[..., SPV5_REFERENCE_CACHE_DIM:],
        None,
        reference_cache=context[..., :SPV5_REFERENCE_CACHE_DIM],
      )
    else:
      estimate = self.estimate_root_state(obs).detach()
      decoded = self.encode_reference(obs).detach()
      features = self._spv5_features(obs, estimate, decoded)
    return torch.cat(
      (self.policy_normalizer(features), self._direct_observations(obs)),
      dim=-1,
    )

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _SPV61DirectActorExport(self)

  def as_jit(self) -> nn.Module:
    return _SPV61DirectActorExport(self)


class SPV61DirectCritic(nn.Module):
  """HEFT critic receiving actual DR parameters and push history directly."""

  is_recurrent = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: Sequence[int] = (1024, 512, 512),
    activation: str = "mish",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    vecnorm_decay: float = 0.9999,
    rma_physics_actual_group: str = "rma_physics_actual",
    rma_push_history_group: str = "rma_push_history",
  ) -> None:
    super().__init__()
    del distribution_cfg
    if activation.lower() != "mish":
      raise ValueError("SPV61DirectCritic requires Mish activation")
    self.obs_groups = list(obs_groups[obs_set])
    self.physics_group = str(rma_physics_actual_group)
    self.push_group = str(rma_push_history_group)
    if self.obs_groups[-2:] != [self.physics_group, self.push_group]:
      raise ValueError(
        "SPV6-1 critic groups must end in actual physics and push history"
      )
    self.base_groups = self.obs_groups[:-2]
    self.base_dim = sum(int(obs[name].shape[-1]) for name in self.base_groups)
    if int(obs[self.physics_group].shape[-1]) != SPV6_PHYSICS_DIM:
      raise ValueError("SPV6-1 physical parameter dimension mismatch")
    if int(obs[self.push_group].shape[-1]) != SPV6_PUSH_HISTORY_DIM:
      raise ValueError("SPV6-1 push history dimension mismatch")
    self.obs_dim = self.base_dim + SPV6_1_DIRECT_DIM
    self.obs_normalization = bool(obs_normalization)
    self.base_normalizer = (
      DecayVecNorm(self.base_dim, decay=vecnorm_decay)
      if self.obs_normalization else nn.Identity()
    )
    self.mlp = _make_mlp(self.obs_dim, hidden_dims, output_dim)
    self.apply(_orthogonal_small_)

  def _base(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs[name] for name in self.base_groups], dim=-1)

  def get_latent(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat(
      (
        self.base_normalizer(self._base(obs)),
        obs[self.physics_group],
        obs[self.push_group],
      ),
      dim=-1,
    )

  def forward(
    self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output=False
  ) -> torch.Tensor:
    del hidden_state, stochastic_output
    obs = unpad_trajectories(obs, masks) if masks is not None else obs
    return self.mlp(self.get_latent(obs))

  @torch.no_grad()
  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      self.base_normalizer.update(self._base(obs))  # type: ignore[attr-defined]

  def reset(self, dones=None, hidden_state=None) -> None:
    del dones, hidden_state

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None) -> None:
    del dones

  def adamw_only_parameters(self):
    return self.mlp[-1].parameters()


class _SPV61DirectActorExport(_SPV5ActorExport):
  def __init__(self, model: SPV61DirectActor) -> None:
    super().__init__(model)

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    quat_end = SPV5_ROBOT_ROOT_QUAT_DIM
    history_end = quat_end + self.history_length * sum(PROPRIO_TERM_DIMS)
    reference_end = history_end + SPV5_REFERENCE_INPUT_DIM
    key_body_end = reference_end + SPV4_KEY_BODY_STATE_DIM
    robot_root_quat = value[..., :quat_end]
    history = value[..., quat_end:history_end]
    reference_input = value[..., history_end:reference_end]
    robot_key_body = value[..., reference_end:key_body_end]
    direct = value[..., key_body_end:]

    estimate = self.estimator(self.history_normalizer(history))
    residual_normalized = self.reference_encoder(
      self.reference_input_normalizer(reference_input)
    )
    decoded = reference_input[..., -SPV5_REFERENCE_TARGET_DIM:] + (
      _normalizer_inverse(self.reference_residual_normalizer, residual_normalized)
    )
    features = _spv5_policy_features(
      history=history,
      latest_proprio=self._latest_policy_proprio(history),
      estimate=estimate,
      decoded_reference=decoded,
      robot_root_quat=robot_root_quat,
      robot_key_body=robot_key_body,
      history_length=self.history_length,
      kinematics=self.reference_kinematics,
    )
    return self.deterministic_output(
      self.mlp(torch.cat((self.policy_normalizer(features), direct), dim=-1))
    )

  @property
  def input_names(self) -> list[str]:
    return ["spv6_1_observation"]

  @property
  def deploy_input_names(self) -> list[str]:
    return ["spv6_1_observation"]
