"""HEFT teacher-only models used by the SP pretrain task."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from .residual_moe import ObservationConditionedResidualMoE


def _make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  current = int(input_dim)
  for width in hidden_dims:
    width = int(width)
    layers.extend((nn.Linear(current, width), nn.LayerNorm(width), nn.Mish()))
    current = width
  layers.append(nn.Linear(current, int(output_dim)))
  return nn.Sequential(*layers)


def _orthogonal_small_(module: nn.Module) -> None:
  if isinstance(module, nn.Linear):
    nn.init.orthogonal_(module.weight, gain=0.01)
    nn.init.zeros_(module.bias)


class DecayVecNorm(nn.Module):
  """TorchRL VecNorm-compatible discounted running normalization."""

  def __init__(self, size: int, decay: float = 0.9999, eps: float = 1.0e-6):
    super().__init__()
    self.decay = float(decay)
    self.eps = float(eps)
    self.register_buffer("sum", torch.zeros(size))
    self.register_buffer("ssq", torch.zeros(size))
    self.register_buffer("count", torch.zeros(1))

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    count = self.count.clamp_min(1.0)
    mean = self.sum / count
    variance = (self.ssq / count - mean.square()).clamp_min(self.eps)
    return (value - mean) / variance.sqrt().clamp_min(self.eps)

  @torch.no_grad()
  def update(self, value: torch.Tensor) -> None:
    if not self.training:
      return
    flat = value.reshape(-1, value.shape[-1])
    self.sum.mul_(self.decay).add_(flat.sum(dim=0))
    self.ssq.mul_(self.decay).add_(flat.square().sum(dim=0))
    self.count.mul_(self.decay).add_(float(flat.shape[0]))


class _HeftModelBase(nn.Module):
  is_recurrent = False

  def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  def _flat_obs(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs[name] for name in self.obs_groups], dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    self.obs_normalizer.update(self._flat_obs(obs))


class HeftTeacherActor(_HeftModelBase):
  """HEFT pretrain teacher: policy plus an encoded privileged observation."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: Sequence[int] = (1024, 1024, 512),
    activation: str = "mish",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    privileged_latent_dim: int = 256,
    vecnorm_decay: float = 0.9999,
    init_std: Sequence[float] | None = None,
  ) -> None:
    super().__init__()
    if activation.lower() != "mish":
      raise ValueError("HeftTeacherActor requires Mish activation")
    self.obs_groups = list(obs_groups[obs_set])
    if self.obs_groups != ["policy", "priv"]:
      raise ValueError(
        "HeftTeacherActor requires actor observation groups [policy, priv], "
        f"got {self.obs_groups}"
      )
    self.policy_dim = int(obs["policy"].shape[-1])
    self.priv_dim = int(obs["priv"].shape[-1])
    self.obs_dim = self.policy_dim + self.priv_dim
    self.obs_normalization = bool(obs_normalization)
    self.obs_normalizer = (
      DecayVecNorm(self.obs_dim, decay=vecnorm_decay)
      if self.obs_normalization
      else nn.Identity()
    )
    latent_dim = int(privileged_latent_dim)
    self.encoder_priv = _make_mlp(self.priv_dim, (512,), latent_dim)
    self.mlp = _make_mlp(self.policy_dim + latent_dim, hidden_dims, output_dim)
    cfg = dict(distribution_cfg or {})
    cfg.pop("class_name", None)
    cfg_init_std = float(cfg.pop("init_std", 1.0))
    self.distribution = GaussianDistribution(output_dim, init_std=cfg_init_std, **cfg)
    if init_std is not None:
      resolved = torch.as_tensor(init_std, dtype=torch.float32)
      if resolved.numel() != output_dim:
        raise ValueError(f"init_std has {resolved.numel()} values, expected {output_dim}")
      self.distribution.std_param.data.copy_(resolved)
      self._std_upper = resolved.clone()
    else:
      self._std_upper = torch.full((output_dim,), cfg_init_std)
    self.apply(_orthogonal_small_)

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    del masks, hidden_state
    flat = self._flat_obs(obs)
    flat = self.obs_normalizer(flat)
    policy, priv = flat.split((self.policy_dim, self.priv_dim), dim=-1)
    return torch.cat((policy, self.encoder_priv(priv)), dim=-1)

  def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output=False):
    obs = unpad_trajectories(obs, masks) if masks is not None else obs
    output = self.mlp(self.get_latent(obs, hidden_state=hidden_state))
    if stochastic_output:
      self.distribution.update(output)
      return self.distribution.sample()
    return output

  @property
  def output_mean(self):
    return self.distribution.mean

  @property
  def output_std(self):
    return self.distribution.std

  @property
  def output_entropy(self):
    return self.distribution.entropy

  @property
  def output_distribution_params(self):
    return self.distribution.params

  def get_output_log_prob(self, outputs):
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(self, old_params, new_params):
    return self.distribution.kl_divergence(old_params, new_params)

  @torch.no_grad()
  def clamp_std(self) -> None:
    std = self.distribution.std_param
    upper = self._std_upper.to(device=std.device, dtype=std.dtype)
    std.copy_(torch.minimum(std, upper))

  def adamw_only_parameters(self):
    return self.mlp[-1].parameters()

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _HeftActorExport(self)

  def as_jit(self) -> nn.Module:
    return _HeftActorExport(self)


class HeftTeacherCritic(_HeftModelBase):
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
  ) -> None:
    super().__init__()
    del distribution_cfg
    if activation.lower() != "mish":
      raise ValueError("HeftTeacherCritic requires Mish activation")
    self.obs_groups = list(obs_groups[obs_set])
    self.obs_dim = sum(int(obs[name].shape[-1]) for name in self.obs_groups)
    self.obs_normalization = bool(obs_normalization)
    self.obs_normalizer = (
      DecayVecNorm(self.obs_dim, decay=vecnorm_decay)
      if self.obs_normalization
      else nn.Identity()
    )
    self.mlp = _make_mlp(self.obs_dim, hidden_dims, output_dim)
    self.apply(_orthogonal_small_)

  def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output=False):
    del hidden_state, stochastic_output
    obs = unpad_trajectories(obs, masks) if masks is not None else obs
    return self.mlp(self.obs_normalizer(self._flat_obs(obs)))

  def adamw_only_parameters(self):
    return self.mlp[-1].parameters()


class HeftTeacherMoECritic(HeftTeacherCritic):
  """HEFT critic whose value MLP is replaced by a residual MoE core."""

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
    moe_context_hidden_dim: int = 1472,
    moe_hidden_dim: int = 608,
    moe_num_experts: int = 8,
    moe_top_k: int = 2,
    moe_expansion: int = 4,
    moe_router_temperature: float = 1.5,
    moe_router_init_std: float = 1.0e-2,
    moe_output_init_gain: float = 1.0e-2,
  ) -> None:
    del hidden_dims
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=(),
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
      vecnorm_decay=vecnorm_decay,
    )
    self.mlp = ObservationConditionedResidualMoE(
      self.obs_dim,
      output_dim,
      context_hidden_dim=moe_context_hidden_dim,
      hidden_dim=moe_hidden_dim,
      num_experts=moe_num_experts,
      top_k=moe_top_k,
      expansion=moe_expansion,
      router_temperature=moe_router_temperature,
      router_init_std=moe_router_init_std,
      output_init_gain=moe_output_init_gain,
    )

  def routing_probabilities(self, obs: TensorDict) -> torch.Tensor:
    """Return dense probabilities over all experts for routing losses."""
    normalized = self.obs_normalizer(self._flat_obs(obs))
    return self.mlp.routing_probabilities(normalized)

  @property
  def moe_value_parameter_count(self) -> int:
    return self.mlp.dense_parameter_count

  def adamw_only_parameters(self):
    return self.mlp.output.parameters()


class _HeftActorExport(nn.Module):
  def __init__(self, model: HeftTeacherActor):
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.encoder_priv = copy.deepcopy(model.encoder_priv)
    self.mlp = copy.deepcopy(model.mlp)
    self.policy_dim = model.policy_dim
    self.priv_dim = model.priv_dim
    self.input_size = model.obs_dim

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    value = self.obs_normalizer(value)
    policy, priv = value.split((self.policy_dim, self.priv_dim), dim=-1)
    return self.mlp(torch.cat((policy, self.encoder_priv(priv)), dim=-1))

  def get_dummy_inputs(self):
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self):
    return ["policy_priv"]

  @property
  def deploy_input_names(self):
    return ["policy_priv"]

  @property
  def output_names(self):
    return ["action"]
