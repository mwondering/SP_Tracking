"""Periodic-autoencoder routed mixture-of-experts building blocks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)

from .residual_moe import LayerNormResidualBlock, RMSNorm
from .spv3_models import _identity_or_normalizer


def _normalize_vector(value: torch.Tensor) -> torch.Tensor:
  return value / torch.linalg.vector_norm(
    value, dim=-1, keepdim=True
  ).clamp_min(1.0e-9)


def _rot6d_to_quat(value: torch.Tensor) -> torch.Tensor:
  """TorchScript-compatible 6D rotation conversion for the PMoE export."""
  first = _normalize_vector(value[..., :3])
  second_raw = value[..., 3:6]
  second = _normalize_vector(
    second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first
  )
  third = torch.linalg.cross(first, second, dim=-1)
  matrix = torch.stack((first, second, third), dim=-1)
  return _normalize_vector(quat_from_matrix(matrix))


def _quat_to_rot6d(value: torch.Tensor) -> torch.Tensor:
  return (
    matrix_from_quat(value)[..., :, :2]
    .transpose(-2, -1)
    .flatten(start_dim=-2)
  )


class PeriodicAutoencoder(nn.Module):
  """DeepPhase-style periodic bottleneck for fixed reference windows.

  The explicit DFT matrices are equivalent to the positive-frequency part of
  ``torch.fft.rfft`` while remaining friendly to the legacy ONNX exporter.
  """

  def __init__(
    self,
    input_dim: int,
    latent_dim: int,
    window_length: int,
    window_seconds: float,
    *,
    hidden_dims: Sequence[int] = (64, 64),
    kernel_size: int = 5,
  ) -> None:
    super().__init__()
    input_dim = int(input_dim)
    latent_dim = int(latent_dim)
    window_length = int(window_length)
    kernel_size = int(kernel_size)
    widths = tuple(int(value) for value in hidden_dims)
    if min(input_dim, latent_dim, window_length) <= 0:
      raise ValueError("PAE dimensions must be positive")
    if window_seconds <= 0.0:
      raise ValueError("PAE window_seconds must be positive")
    if not widths or min(widths) <= 0:
      raise ValueError("PAE hidden dimensions must be positive")
    if kernel_size <= 0 or kernel_size % 2 == 0:
      raise ValueError("PAE kernel_size must be a positive odd number")

    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.window_length = window_length
    self.window_seconds = float(window_seconds)
    padding = kernel_size // 2

    encoder_layers: list[nn.Module] = []
    channels = (input_dim, *widths, latent_dim)
    for input_channels, output_channels in zip(
      channels[:-1], channels[1:], strict=True
    ):
      encoder_layers.extend(
        (
          nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=padding,
          ),
          nn.ELU(),
        )
      )
    self.encoder = nn.Sequential(*encoder_layers)

    self.phase_encoders = nn.ModuleList(
      nn.Linear(window_length, 2) for _ in range(latent_dim)
    )

    decoder_layers: list[nn.Module] = []
    decoder_channels = (latent_dim, *reversed(widths), input_dim)
    for index, (input_channels, output_channels) in enumerate(
      zip(decoder_channels[:-1], decoder_channels[1:], strict=True)
    ):
      decoder_layers.append(
        nn.Conv1d(
          input_channels,
          output_channels,
          kernel_size,
          padding=padding,
        )
      )
      if index + 1 < len(decoder_channels) - 1:
        decoder_layers.append(nn.ELU())
    self.decoder = nn.Sequential(*decoder_layers)

    frequency_indices = torch.arange(
      1, window_length // 2 + 1, dtype=torch.float32
    )
    sample_indices = torch.arange(window_length, dtype=torch.float32)
    angle = (
      2.0
      * math.pi
      * frequency_indices.unsqueeze(1)
      * sample_indices.unsqueeze(0)
      / float(window_length)
    )
    self.register_buffer("dft_cos", torch.cos(angle), persistent=False)
    self.register_buffer("dft_sin", torch.sin(angle), persistent=False)
    self.register_buffer(
      "frequency_hz",
      frequency_indices / float(window_seconds),
      persistent=False,
    )
    self.register_buffer(
      "time_seconds",
      torch.linspace(
        -float(window_seconds) / 2.0,
        float(window_seconds) / 2.0,
        window_length,
      ),
      persistent=False,
    )

  def encode(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
    if (
      value.ndim != 3
      or value.shape[1] != self.input_dim
      or value.shape[2] != self.window_length
    ):
      raise ValueError("PAE input has an invalid shape")
    latent = self.encoder(value)
    real = torch.matmul(latent, self.dft_cos.transpose(0, 1))
    imaginary = -torch.matmul(latent, self.dft_sin.transpose(0, 1))
    power = real.square() + imaginary.square()
    power_sum = power.sum(dim=-1).clamp_min(1.0e-8)
    amplitude = (
      2.0 * torch.sqrt(power_sum) / float(self.window_length)
    )
    frequency = (
      power * self.frequency_hz
    ).sum(dim=-1) / power_sum
    offset = latent.mean(dim=-1)

    phase_vectors = torch.stack(
      [
        phase_encoder(latent[:, channel_index])
        for channel_index, phase_encoder in enumerate(self.phase_encoders)
      ],
      dim=1,
    )
    phase_shift = torch.atan2(
      phase_vectors[..., 1], phase_vectors[..., 0]
    ) / (2.0 * math.pi)
    return {
      "latent": latent,
      "amplitude": amplitude,
      "frequency": frequency,
      "offset": offset,
      "phase_shift": phase_shift,
    }

  def decode_parameters(
    self, parameters: dict[str, torch.Tensor]
  ) -> torch.Tensor:
    amplitude = parameters["amplitude"].unsqueeze(-1)
    frequency = parameters["frequency"].unsqueeze(-1)
    offset = parameters["offset"].unsqueeze(-1)
    phase_shift = parameters["phase_shift"].unsqueeze(-1)
    harmonic_latent = amplitude * torch.sin(
      2.0
      * math.pi
      * (
        frequency * self.time_seconds.reshape(1, 1, -1)
        + phase_shift
      )
    ) + offset
    return self.decoder(harmonic_latent)

  def forward(
    self, value: torch.Tensor
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parameters = self.encode(value)
    return self.decode_parameters(parameters), parameters

  def cluster_embedding(self, value: torch.Tensor) -> torch.Tensor:
    """Return phase-invariant window features used by online K-Means."""
    parameters = self.encode(value)
    return torch.cat(
      (
        torch.log1p(parameters["amplitude"]),
        parameters["frequency"],
        parameters["offset"],
      ),
      dim=-1,
    )


class OnlineKMeansRouter(nn.Module):
  """No-gradient online K-Means with soft nearest-prototype routing."""

  def __init__(
    self,
    feature_dim: int,
    num_clusters: int,
    *,
    temperature: float = 1.0,
    momentum: float = 0.99,
    eps: float = 1.0e-6,
  ) -> None:
    super().__init__()
    feature_dim = int(feature_dim)
    num_clusters = int(num_clusters)
    if feature_dim <= 0 or num_clusters <= 1:
      raise ValueError("K-Means dimensions must be positive")
    if temperature <= 0.0:
      raise ValueError("K-Means temperature must be positive")
    if not 0.0 <= momentum < 1.0:
      raise ValueError("K-Means momentum must be in [0, 1)")
    self.feature_dim = feature_dim
    self.num_clusters = num_clusters
    self.temperature = float(temperature)
    self.momentum = float(momentum)
    self.eps = float(eps)

    self.register_buffer(
      "centroids", torch.zeros(num_clusters, feature_dim)
    )
    self.register_buffer("feature_mean", torch.zeros(feature_dim))
    self.register_buffer("feature_var", torch.ones(feature_dim))
    self.register_buffer(
      "assignment_counts", torch.zeros(num_clusters, dtype=torch.float64)
    )
    self.register_buffer(
      "initialized", torch.tensor(False, dtype=torch.bool)
    )
    self.register_buffer(
      "statistics_initialized", torch.tensor(False, dtype=torch.bool)
    )
    self.register_buffer(
      "num_updates", torch.tensor(0, dtype=torch.long)
    )

  def standardized_distances(self, features: torch.Tensor) -> torch.Tensor:
    scale = torch.sqrt(self.feature_var.clamp_min(self.eps))
    delta = (
      features.unsqueeze(-2) - self.centroids
    ) / scale
    return delta.square().mean(dim=-1)

  def probabilities(self, features: torch.Tensor) -> torch.Tensor:
    if not bool(self.initialized.item()):
      return features.new_full(
        (features.shape[0], self.num_clusters),
        1.0 / float(self.num_clusters),
      )
    return torch.softmax(
      -self.standardized_distances(features) / self.temperature,
      dim=-1,
    )

  def hard_assignments(self, features: torch.Tensor) -> torch.Tensor:
    if not bool(self.initialized.item()):
      raise RuntimeError("K-Means centroids have not been initialized")
    return self.standardized_distances(features).argmin(dim=-1)

  @torch.no_grad()
  def initialize(self, features: torch.Tensor) -> None:
    flat = features.reshape(-1, self.feature_dim)
    if flat.shape[0] == 0:
      raise ValueError("Cannot initialize K-Means from an empty feature batch")
    scale = torch.sqrt(self.feature_var.clamp_min(self.eps))
    normalized = (flat - self.feature_mean) / scale
    first_index = normalized.square().sum(dim=-1).argmax()
    selected = [flat[first_index]]
    minimum_distance = (
      (normalized - normalized[first_index]).square().mean(dim=-1)
    )
    for _ in range(1, self.num_clusters):
      next_index = minimum_distance.argmax()
      selected.append(flat[next_index])
      next_distance = (
        (normalized - normalized[next_index]).square().mean(dim=-1)
      )
      minimum_distance = torch.minimum(minimum_distance, next_distance)
    self.centroids.copy_(torch.stack(selected))
    self.initialized.fill_(True)

  @torch.no_grad()
  def update_feature_statistics(
    self,
    feature_sum: torch.Tensor,
    feature_square_sum: torch.Tensor,
    count: float,
  ) -> None:
    if count <= 0.0:
      raise ValueError("K-Means feature count must be positive")
    batch_mean = feature_sum / float(count)
    batch_var = (
      feature_square_sum / float(count) - batch_mean.square()
    ).clamp_min(self.eps)
    if not bool(self.statistics_initialized.item()):
      self.feature_mean.copy_(batch_mean)
      self.feature_var.copy_(batch_var)
      self.statistics_initialized.fill_(True)
      return
    self.feature_mean.lerp_(batch_mean, 1.0 - self.momentum)
    self.feature_var.lerp_(batch_var, 1.0 - self.momentum)

  @torch.no_grad()
  def update_centroids(
    self,
    cluster_sums: torch.Tensor,
    cluster_counts: torch.Tensor,
  ) -> None:
    if cluster_sums.shape != self.centroids.shape:
      raise ValueError("K-Means cluster sums have the wrong shape")
    if cluster_counts.shape != (self.num_clusters,):
      raise ValueError("K-Means cluster counts have the wrong shape")
    nonempty = cluster_counts > 0
    means = cluster_sums / cluster_counts.clamp_min(1.0).unsqueeze(-1)
    if int(self.num_updates.item()) == 0:
      self.centroids[nonempty] = means[nonempty]
    else:
      self.centroids[nonempty] = torch.lerp(
        self.centroids[nonempty],
        means[nonempty],
        1.0 - self.momentum,
      )
    self.assignment_counts.add_(
      cluster_counts.to(dtype=self.assignment_counts.dtype)
    )
    self.num_updates.add_(1)


class PMoERoutingEncoder(nn.Module):
  """Canonicalized PAE features followed by detached online K-Means."""

  def __init__(
    self,
    *,
    frame_dim: int,
    window_length: int,
    reference_fps: float,
    latent_dim: int,
    num_clusters: int,
    hidden_dims: Sequence[int] = (64, 64),
    kernel_size: int = 5,
    cluster_temperature: float = 1.0,
    cluster_momentum: float = 0.99,
    input_normalization: bool = True,
  ) -> None:
    super().__init__()
    self.frame_dim = int(frame_dim)
    self.window_length = int(window_length)
    self.reference_input_dim = self.frame_dim * self.window_length
    if self.frame_dim < 10:
      raise ValueError("PMoE reference frames must contain root pose and joints")
    self.input_normalizer = _identity_or_normalizer(
      bool(input_normalization), self.frame_dim
    )
    self.pae = PeriodicAutoencoder(
      self.frame_dim,
      int(latent_dim),
      self.window_length,
      self.window_length / float(reference_fps),
      hidden_dims=hidden_dims,
      kernel_size=kernel_size,
    )
    self.clusterer = OnlineKMeansRouter(
      3 * int(latent_dim),
      int(num_clusters),
      temperature=cluster_temperature,
      momentum=cluster_momentum,
    )

  def canonicalize(self, reference_input: torch.Tensor) -> torch.Tensor:
    if reference_input.ndim != 2:
      raise ValueError("PMoE reference input must be a two-dimensional batch")
    if reference_input.shape[-1] != self.reference_input_dim:
      raise ValueError(
        f"PMoE reference has {reference_input.shape[-1]} values, expected "
        f"{self.reference_input_dim}"
      )
    frames = reference_input.reshape(
      -1, self.window_length, self.frame_dim
    )
    root_pos = frames[..., :3]
    root_quat = _rot6d_to_quat(frames[..., 3:9])
    first_pos = root_pos[:, :1]
    xy_origin = torch.cat(
      (first_pos[..., :2], torch.zeros_like(first_pos[..., 2:3])),
      dim=-1,
    )
    first_heading = yaw_quat(root_quat[:, :1])
    relative_pos = quat_apply_inverse(
      first_heading.expand(-1, self.window_length, -1),
      root_pos - xy_origin,
    )
    relative_quat = quat_mul(
      quat_conjugate(
        first_heading.expand(-1, self.window_length, -1)
      ),
      root_quat,
    )
    canonical = torch.cat(
      (
        relative_pos,
        _quat_to_rot6d(relative_quat),
        frames[..., 9:],
      ),
      dim=-1,
    )
    return canonical

  def normalized_window(self, reference_input: torch.Tensor) -> torch.Tensor:
    canonical = self.canonicalize(reference_input)
    normalized = self.input_normalizer(canonical)
    return normalized.transpose(1, 2)

  def embeddings(self, reference_input: torch.Tensor) -> torch.Tensor:
    return self.pae.cluster_embedding(
      self.normalized_window(reference_input)
    )

  def forward(self, reference_input: torch.Tensor) -> torch.Tensor:
    return self.clusterer.probabilities(self.embeddings(reference_input))

  def reconstruction_loss(
    self, reference_input: torch.Tensor
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = self.normalized_window(reference_input)
    prediction, parameters = self.pae(target)
    total_mse = F.mse_loss(prediction, target)
    prediction_frames = prediction.transpose(1, 2)
    target_frames = target.transpose(1, 2)
    amplitude = parameters["amplitude"]
    return total_mse, {
      "pmoe_pae_mse": total_mse,
      "pmoe_pae_root_mse": F.mse_loss(
        prediction_frames[..., :9], target_frames[..., :9]
      ),
      "pmoe_pae_joint_mse": F.mse_loss(
        prediction_frames[..., 9:], target_frames[..., 9:]
      ),
      "pmoe_pae_mean_amplitude": amplitude.mean(),
      "pmoe_pae_active_channel_fraction": (
        amplitude > 1.0e-3
      ).float().mean(),
    }

  @torch.no_grad()
  def update_normalization(self, reference_input: torch.Tensor) -> None:
    update = getattr(self.input_normalizer, "update", None)
    if callable(update):
      canonical = self.canonicalize(reference_input)
      update(canonical.reshape(-1, self.frame_dim))


class PrototypeRoutedResidualMoE(nn.Module):
  """Residual MoE whose detached routes come from PAE K-Means prototypes."""

  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    *,
    context_hidden_dim: int = 1472,
    hidden_dim: int = 608,
    num_experts: int = 8,
    top_k: int = 2,
    expansion: int = 4,
    output_init_gain: float = 5.0e-2,
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
      raise ValueError("PMoE dimensions must be positive")
    if num_experts <= 1:
      raise ValueError("PMoE num_experts must be greater than one")
    if not 1 <= top_k <= num_experts:
      raise ValueError("PMoE top_k must be between one and num_experts")

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.num_experts = num_experts
    self.top_k = top_k
    self.context_encoder = nn.Sequential(
      nn.Linear(input_dim, context_hidden_dim),
      nn.ReLU(),
      nn.Linear(context_hidden_dim, hidden_dim),
      nn.ReLU(),
    )
    self.shared_block = LayerNormResidualBlock(hidden_dim, expansion)
    self.experts = nn.ModuleList(
      LayerNormResidualBlock(hidden_dim, expansion)
      for _ in range(num_experts)
    )
    self.post_norm = RMSNorm(hidden_dim)
    self.output = nn.Linear(hidden_dim, output_dim)
    nn.init.orthogonal_(self.output.weight, gain=float(output_init_gain))
    nn.init.zeros_(self.output.bias)

  def sparse_probabilities(
    self, dense_probabilities: torch.Tensor
  ) -> torch.Tensor:
    top_values, top_indices = torch.topk(
      dense_probabilities, self.top_k, dim=-1
    )
    top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(
      1.0e-8
    )
    return torch.zeros_like(dense_probabilities).scatter(
      -1, top_indices, top_values
    )

  def forward(
    self,
    value: torch.Tensor,
    routing_probabilities: torch.Tensor,
  ) -> torch.Tensor:
    if routing_probabilities.shape[:-1] != value.shape[:-1]:
      raise ValueError("PMoE routes and policy features have different batches")
    if routing_probabilities.shape[-1] != self.num_experts:
      raise ValueError(
        f"PMoE route has {routing_probabilities.shape[-1]} experts, expected "
        f"{self.num_experts}"
      )
    shared = self.shared_block(
      self.context_encoder(value)
    )
    # This detach is the algorithmic boundary: PPO may train experts and the
    # action head, but never the PAE or prototype assignments.
    dense_routes = routing_probabilities.detach()
    sparse_routes = self.sparse_probabilities(dense_routes)
    expert_residuals = torch.stack(
      [expert.residual(shared) for expert in self.experts], dim=-2
    )
    mixed_residual = torch.sum(
      sparse_routes.unsqueeze(-1) * expert_residuals, dim=-2
    )
    output = self.output(self.post_norm(shared + mixed_residual))
    return output

  @property
  @torch.jit.unused
  def dense_parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self.parameters())
