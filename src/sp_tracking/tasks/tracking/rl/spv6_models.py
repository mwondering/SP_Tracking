"""SPV6 RMA actor, privileged critic, and anti-collapse decoders."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import HiddenState, MLP
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from sp_tracking.tasks.tracking.mdp.spv4 import SPV4_KEY_BODY_STATE_DIM
from sp_tracking.tasks.tracking.mdp.spv5 import (
  SPV5_REFERENCE_INPUT_DIM,
  SPV5_REFERENCE_TARGET_DIM,
  SPV5_ROBOT_ROOT_QUAT_DIM,
)
from sp_tracking.tasks.tracking.mdp.spv6 import (
  SPV6_GLOBAL_PHYSICS_DIM,
  SPV6_PHYSICS_DIM,
  SPV6_PUSH_FRAME_DIM,
  SPV6_PUSH_HISTORY_DIM,
  SPV6_PUSH_HISTORY_LENGTH,
  SPV6_SENSOR_PHYSICS_DIM,
)

from .heft_models import DecayVecNorm, _make_mlp, _orthogonal_small_
from .spv3_models import (
  PROPRIO_TERM_DIMS,
  SPV3_ESTIMATOR_OUTPUT_DIM,
  _identity_or_normalizer,
)
from .spv5_models import (
  SPV5_POLICY_INPUT_DIM,
  SPV5_REFERENCE_CACHE_DIM,
  SPV5ReferenceEncoderActor,
  _SPV5ActorExport,
  _normalizer_inverse,
  _spv5_policy_features,
)


SPV6_GLOBAL_LATENT_DIM = 8
SPV6_SENSOR_LATENT_DIM = 32
SPV6_PUSH_LATENT_DIM = 16
SPV6_RMA_LATENT_DIM = (
  SPV6_GLOBAL_LATENT_DIM + SPV6_SENSOR_LATENT_DIM + SPV6_PUSH_LATENT_DIM
)


class _FlatHistoryBackbone(nn.Module):
  """Memory-bounded flat encoder for actor and critic history windows."""

  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 256):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ELU(),
      nn.Linear(hidden_dim, output_dim),
      nn.ELU(),
    )

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    return self.network(history)


class SPV6RmaActor(SPV5ReferenceEncoderActor):
  """Deployable SPV5 actor with RMA latents inferred from 50-frame proprioception."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    *,
    rma_physics_nominal_group: str = "rma_physics_nominal",
    rma_global_latent_dim: int = SPV6_GLOBAL_LATENT_DIM,
    rma_sensor_latent_dim: int = SPV6_SENSOR_LATENT_DIM,
    rma_push_latent_dim: int = SPV6_PUSH_LATENT_DIM,
    **kwargs,
  ) -> None:
    self.rma_physics_nominal_group = str(rma_physics_nominal_group)
    self.rma_global_latent_dim = int(rma_global_latent_dim)
    self.rma_sensor_latent_dim = int(rma_sensor_latent_dim)
    self.rma_push_latent_dim = int(rma_push_latent_dim)
    self.rma_latent_dim = (
      self.rma_global_latent_dim
      + self.rma_sensor_latent_dim
      + self.rma_push_latent_dim
    )
    if self.rma_latent_dim != SPV6_RMA_LATENT_DIM:
      raise ValueError(
        f"SPV6 requires {SPV6_RMA_LATENT_DIM} RMA latent values, "
        f"got {self.rma_latent_dim}"
      )
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      extra_actor_groups=(self.rma_physics_nominal_group,),
      extra_policy_input_dim=self.rma_latent_dim,
      raw_actor_obs_extra_dim=SPV6_PHYSICS_DIM,
      **kwargs,
    )
    nominal_dim = int(obs[self.rma_physics_nominal_group].shape[-1])
    if nominal_dim != SPV6_PHYSICS_DIM:
      raise ValueError(
        f"SPV6 nominal physics has {nominal_dim} values, "
        f"expected {SPV6_PHYSICS_DIM}"
      )
    # SPV5 normalizes policy features only.  RMA latents stay in their raw
    # tanh-bounded space and therefore need neither statistics nor a rollout
    # encoder pass merely to update those statistics.
    self.policy_normalizer = _identity_or_normalizer(
      self.obs_normalization, SPV5_POLICY_INPUT_DIM
    )
    self.rma_history_backbone = _FlatHistoryBackbone(
      self.estimator_history_length * sum(PROPRIO_TERM_DIMS), 512
    )
    self.rma_nominal_encoder = nn.Sequential(
      nn.Linear(SPV6_PHYSICS_DIM, 128),
      nn.ELU(),
      nn.Linear(128, 64),
      nn.ELU(),
    )
    fused_dim = 256 + 64
    self.rma_global_head = nn.Linear(fused_dim, self.rma_global_latent_dim)
    self.rma_sensor_head = nn.Linear(fused_dim, self.rma_sensor_latent_dim)
    self.rma_push_head = nn.Linear(fused_dim, self.rma_push_latent_dim)
    for head in (self.rma_global_head, self.rma_sensor_head, self.rma_push_head):
      nn.init.orthogonal_(head.weight, gain=0.01)
      nn.init.zeros_(head.bias)

  def rma_latents(
    self, obs: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_history = obs[self.estimator_history_group]
    history = self.history_normalizer(raw_history)
    if torch.is_grad_enabled():
      self._cached_normalized_history = (raw_history, history)
    history_feature = self.rma_history_backbone(history)
    # Physical calibration parameters deliberately bypass observation normalization.
    nominal_feature = self.rma_nominal_encoder(
      obs[self.rma_physics_nominal_group]
    )
    fused = torch.cat((history_feature, nominal_feature), dim=-1)
    return (
      torch.tanh(self.rma_global_head(fused)),
      torch.tanh(self.rma_sensor_head(fused)),
      torch.tanh(self.rma_push_head(fused)),
    )

  def estimator_losses(
    self, obs: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor]:
    cache = getattr(self, "_cached_normalized_history", None)
    self._cached_normalized_history = None
    raw_history = obs[self.estimator_history_group]
    normalized_history = None
    if cache is not None:
      cached_raw, cached_normalized = cache
      if (
        cached_raw.shape == raw_history.shape
        and cached_raw.stride() == raw_history.stride()
        and cached_raw.storage_offset() == raw_history.storage_offset()
        and cached_raw.data_ptr() == raw_history.data_ptr()
      ):
        normalized_history = cached_normalized
    if normalized_history is None:
      normalized_history = self.history_normalizer(raw_history)
    estimate = self.estimator(normalized_history)
    target = obs[self.estimator_target_group]
    if target.shape[-1] != SPV3_ESTIMATOR_OUTPUT_DIM:
      raise ValueError(
        "SPV6 estimator target has "
        f"{target.shape[-1]} values, expected {SPV3_ESTIMATOR_OUTPUT_DIM}"
      )
    return (
      (estimate[..., :1] - target[..., :1]).square().mean(),
      (estimate[..., 1:] - target[..., 1:]).square().mean(),
    )

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    context = obs.get("spv5_policy_context_cache")
    if context is not None:
      reference_cache = context[..., :SPV5_REFERENCE_CACHE_DIM]
      estimate = context[..., SPV5_REFERENCE_CACHE_DIM:]
      features = self._spv5_features(
        obs, estimate, None, reference_cache=reference_cache
      )
    else:
      estimate = self.estimate_root_state(obs).detach()
      decoded = self.encode_reference(obs).detach()
      features = self._spv5_features(obs, estimate, decoded)
    rma_latents = self.rma_latents(obs)
    self._cached_rma_latents = rma_latents
    rma = torch.cat(rma_latents, dim=-1)
    return torch.cat((self.policy_normalizer(features), rma), dim=-1)

  def cached_rma_latents(
    self, batch_size: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    cached = getattr(self, "_cached_rma_latents", None)
    if cached is None or any(value.shape[0] < batch_size for value in cached):
      return None
    return tuple(value[:batch_size] for value in cached)  # type: ignore[return-value]

  @torch.no_grad()
  def update_normalization(self, obs: TensorDict) -> None:
    history = obs[self.estimator_history_group]
    reference_input = obs[self.reference_encoder_input_group]
    target = obs[self.reference_encoder_target_group]
    if self.obs_normalization:
      self.history_normalizer.update(history)  # type: ignore[attr-defined]
      self.reference_input_normalizer.update(reference_input)  # type: ignore[attr-defined]
      self.reference_residual_normalizer.update(  # type: ignore[attr-defined]
        target - self._noisy_support(reference_input)
      )
    context = self.populate_policy_context_cache(obs)
    estimate = context[..., SPV5_REFERENCE_CACHE_DIM:]
    features = self._spv5_features(
      obs,
      estimate,
      None,
      reference_cache=context[..., :SPV5_REFERENCE_CACHE_DIM],
    )
    if self.obs_normalization:
      self.policy_normalizer.update(features)  # type: ignore[attr-defined]

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _SPV6ActorExport(self)

  def as_jit(self) -> nn.Module:
    return _SPV6ActorExport(self)


class SPV6RmaCritic(nn.Module):
  """HEFT critic with raw privileged-physics and kick-history encoders."""

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
    rma_global_latent_dim: int = SPV6_GLOBAL_LATENT_DIM,
    rma_sensor_latent_dim: int = SPV6_SENSOR_LATENT_DIM,
    rma_push_latent_dim: int = SPV6_PUSH_LATENT_DIM,
  ) -> None:
    super().__init__()
    del distribution_cfg
    if activation.lower() != "mish":
      raise ValueError("SPV6RmaCritic requires Mish activation")
    self.obs_groups = list(obs_groups[obs_set])
    self.physics_group = str(rma_physics_actual_group)
    self.push_group = str(rma_push_history_group)
    expected_tail = [self.physics_group, self.push_group]
    if self.obs_groups[-2:] != expected_tail:
      raise ValueError(
        f"SPV6 critic groups must end in {expected_tail}, got {self.obs_groups}"
      )
    self.base_groups = self.obs_groups[:-2]
    self.base_dim = sum(int(obs[name].shape[-1]) for name in self.base_groups)
    if int(obs[self.physics_group].shape[-1]) != SPV6_PHYSICS_DIM:
      raise ValueError("SPV6 critic physical parameter dimension mismatch")
    if int(obs[self.push_group].shape[-1]) != SPV6_PUSH_HISTORY_DIM:
      raise ValueError("SPV6 critic push history dimension mismatch")
    self.global_latent_dim = int(rma_global_latent_dim)
    self.sensor_latent_dim = int(rma_sensor_latent_dim)
    self.push_latent_dim = int(rma_push_latent_dim)
    self.rma_latent_dim = (
      self.global_latent_dim + self.sensor_latent_dim + self.push_latent_dim
    )
    self.obs_dim = self.base_dim + SPV6_PHYSICS_DIM + SPV6_PUSH_HISTORY_DIM
    self.obs_normalization = bool(obs_normalization)
    self.base_normalizer = (
      DecayVecNorm(self.base_dim, decay=vecnorm_decay)
      if self.obs_normalization else nn.Identity()
    )
    self.global_encoder = nn.Sequential(
      nn.Linear(SPV6_GLOBAL_PHYSICS_DIM, 64), nn.Mish(),
      nn.Linear(64, 64), nn.Mish(),
      nn.Linear(64, self.global_latent_dim), nn.Tanh(),
    )
    self.sensor_encoder = nn.Sequential(
      nn.Linear(SPV6_SENSOR_PHYSICS_DIM, 128), nn.Mish(),
      nn.Linear(128, 64), nn.Mish(),
      nn.Linear(64, self.sensor_latent_dim), nn.Tanh(),
    )
    self.push_backbone = _FlatHistoryBackbone(
      SPV6_PUSH_HISTORY_DIM, 128
    )
    self.push_head = nn.Sequential(
      nn.Linear(256, self.push_latent_dim), nn.Tanh()
    )
    self.global_decoder = nn.Sequential(
      nn.Linear(self.global_latent_dim, 64), nn.Mish(),
      nn.Linear(64, SPV6_GLOBAL_PHYSICS_DIM),
    )
    self.sensor_decoder = nn.Sequential(
      nn.Linear(self.sensor_latent_dim, 64), nn.Mish(),
      nn.Linear(64, SPV6_SENSOR_PHYSICS_DIM),
    )
    self.push_decoder = nn.Sequential(
      nn.Linear(self.push_latent_dim, 128), nn.Mish(),
      nn.Linear(128, SPV6_PUSH_HISTORY_LENGTH * SPV6_PUSH_FRAME_DIM),
    )
    self.mlp = _make_mlp(
      self.base_dim + self.rma_latent_dim, hidden_dims, output_dim
    )
    self.apply(_orthogonal_small_)

  def _base(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs[name] for name in self.base_groups], dim=-1)

  def rma_latents(
    self, obs: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    physics = obs[self.physics_group]
    return (
      self.global_encoder(physics[..., :SPV6_GLOBAL_PHYSICS_DIM]),
      self.sensor_encoder(physics[..., SPV6_GLOBAL_PHYSICS_DIM:]),
      self.push_head(self.push_backbone(obs[self.push_group])),
    )

  def cached_rma_latents(
    self, batch_size: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    cached = getattr(self, "_cached_rma_latents", None)
    if cached is None or any(value.shape[0] < batch_size for value in cached):
      return None
    return tuple(value[:batch_size] for value in cached)  # type: ignore[return-value]

  def reconstruction_losses(
    self,
    obs: TensorDict,
    latents: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    physics = obs[self.physics_group]
    z_global, z_sensor, z_push = (
      self.rma_latents(obs) if latents is None else latents
    )
    pred_global = self.global_decoder(z_global)
    pred_sensor = self.sensor_decoder(z_sensor)
    tolerances = physics.new_tensor((0.01, 0.01, 0.01, 0.1, 0.1))
    global_scaled_error = (
      pred_global - physics[..., :SPV6_GLOBAL_PHYSICS_DIM]
    ) / tolerances
    global_loss = F.smooth_l1_loss(
      global_scaled_error, torch.zeros_like(global_scaled_error)
    )
    sensor_scaled_error = (
      pred_sensor - physics[..., SPV6_GLOBAL_PHYSICS_DIM:]
    ) / 0.002
    sensor_loss = F.smooth_l1_loss(
      sensor_scaled_error, torch.zeros_like(sensor_scaled_error)
    )
    physics_loss = global_loss + sensor_loss

    target_push = obs[self.push_group].reshape(
      -1, SPV6_PUSH_HISTORY_LENGTH, SPV6_PUSH_FRAME_DIM
    )
    decoded_push = self.push_decoder(z_push).reshape_as(target_push)
    target_mask = target_push[..., 6]
    mask_logits = decoded_push[..., 6]
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
    event_count = target_mask.sum().clamp_min(1.0)
    velocity_error = F.smooth_l1_loss(
      decoded_push[..., :6], target_push[..., :6], reduction="none"
    ).mean(dim=-1)
    velocity_loss = (velocity_error * target_mask).sum() / event_count
    push_loss = mask_loss + velocity_loss
    diagnostics = {
      "rma_reconstruction_global": global_loss,
      "rma_reconstruction_sensor": sensor_loss,
      "rma_push_mask_bce": mask_loss,
      "rma_push_velocity_huber": velocity_loss,
      "rma_global_mae": (
        pred_global - physics[..., :SPV6_GLOBAL_PHYSICS_DIM]
      ).abs().mean(),
      "rma_sensor_mae": (
        pred_sensor - physics[..., SPV6_GLOBAL_PHYSICS_DIM:]
      ).abs().mean(),
    }
    return physics_loss, push_loss, diagnostics

  def forward(
    self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output=False
  ) -> torch.Tensor:
    del hidden_state, stochastic_output
    obs = unpad_trajectories(obs, masks) if masks is not None else obs
    base = self.base_normalizer(self._base(obs))
    rma_latents = self.rma_latents(obs)
    self._cached_rma_latents = rma_latents
    return self.mlp(torch.cat((base, *rma_latents), dim=-1))

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


class _SPV6ActorExport(_SPV5ActorExport):
  def __init__(self, model: SPV6RmaActor) -> None:
    super().__init__(model)
    self.rma_history_backbone = copy.deepcopy(model.rma_history_backbone)
    self.rma_nominal_encoder = copy.deepcopy(model.rma_nominal_encoder)
    self.rma_global_head = copy.deepcopy(model.rma_global_head)
    self.rma_sensor_head = copy.deepcopy(model.rma_sensor_head)
    self.rma_push_head = copy.deepcopy(model.rma_push_head)

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    quat_end = SPV5_ROBOT_ROOT_QUAT_DIM
    history_end = quat_end + self.history_length * sum(PROPRIO_TERM_DIMS)
    reference_end = history_end + SPV5_REFERENCE_INPUT_DIM
    key_body_end = reference_end + SPV4_KEY_BODY_STATE_DIM
    robot_root_quat = value[..., :quat_end]
    history = value[..., quat_end:history_end]
    reference_input = value[..., history_end:reference_end]
    robot_key_body = value[..., reference_end:key_body_end]
    nominal = value[..., key_body_end:]

    normalized_history = self.history_normalizer(history)
    estimate = self.estimator(normalized_history)
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
    fused = torch.cat(
      (
        self.rma_history_backbone(normalized_history),
        self.rma_nominal_encoder(nominal),
      ),
      dim=-1,
    )
    rma = torch.cat(
      (
        torch.tanh(self.rma_global_head(fused)),
        torch.tanh(self.rma_sensor_head(fused)),
        torch.tanh(self.rma_push_head(fused)),
      ),
      dim=-1,
    )
    return self.deterministic_output(
      self.mlp(torch.cat((self.policy_normalizer(features), rma), dim=-1))
    )

  @property
  def input_names(self) -> list[str]:
    return ["spv6_observation"]

  @property
  def deploy_input_names(self) -> list[str]:
    return ["spv6_observation"]
