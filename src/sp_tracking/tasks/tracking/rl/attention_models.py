"""Attention actor models for G1 BFM tracking ablations."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from typing import ClassVar, cast

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn

TERM_DIMS: OrderedDict[str, int] = OrderedDict(
  (
    ("command", 58),
    ("motion_anchor_pos_b", 3),
    ("motion_anchor_ori_b", 6),
    ("body_pos", 42),
    ("body_ori", 84),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("joint_pos", 29),
    ("joint_vel", 29),
    ("actions", 29),
  )
)
PROPRIO_TERMS = ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions")
SPARSETRACK_PROP_TERMS = ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel")
SPARSETRACK_TASK_TERMS = (
  "command",
  "motion_anchor_pos_b",
  "motion_anchor_ori_b",
  "body_pos",
  "body_ori",
)


def _make_causal_mask(length: int) -> torch.Tensor:
  return torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)


def _make_encoder_layer(
  d_model: int,
  num_heads: int,
  ffn_dim: int,
  dropout: float,
  activation: str,
) -> nn.TransformerEncoderLayer:
  return nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=num_heads,
    dim_feedforward=ffn_dim,
    dropout=dropout,
    activation=activation,
    batch_first=True,
    norm_first=True,
  )


class _CrossAttentionBlock(nn.Module):
  def __init__(
    self,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    activation: str,
  ) -> None:
    super().__init__()
    self.q_norm = nn.LayerNorm(d_model)
    self.kv_norm = nn.LayerNorm(d_model)
    self.attn = nn.MultiheadAttention(
      d_model,
      num_heads,
      dropout=dropout,
      batch_first=True,
    )
    self.ffn_norm = nn.LayerNorm(d_model)
    activation_mod: nn.Module
    if activation == "gelu":
      activation_mod = nn.GELU()
    elif activation == "elu":
      activation_mod = nn.ELU()
    else:
      raise ValueError(f"Unsupported cross-attention activation: {activation}")
    self.ffn = nn.Sequential(
      nn.Linear(d_model, ffn_dim),
      activation_mod,
      nn.Dropout(dropout),
      nn.Linear(ffn_dim, d_model),
      nn.Dropout(dropout),
    )

  def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
    norm_kv = self.kv_norm(key_value)
    attn_out, _ = self.attn(
      self.q_norm(query),
      norm_kv,
      norm_kv,
      need_weights=False,
    )
    query = query + attn_out
    return query + self.ffn(self.ffn_norm(query))


class _RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-8) -> None:
    super().__init__()
    self.scale = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
    return self.scale * x / (norm + self.eps)


class _RoPEPositionalEncoding(nn.Module):
  def __init__(self, dim: int, base: float = 10000.0) -> None:
    super().__init__()
    if dim % 2 != 0:
      raise ValueError("RoPE dimension must be even")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    freqs = torch.outer(position, self.inv_freq)
    cos_freqs = torch.cos(freqs).unsqueeze(0)
    sin_freqs = torch.sin(freqs).unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = x_even * cos_freqs - x_odd * sin_freqs
    x_rotated[..., 1::2] = x_even * sin_freqs + x_odd * cos_freqs
    return x_rotated


class _SwiGLU(nn.Module):
  def __init__(self, input_dim: int, hidden_dim: int) -> None:
    super().__init__()
    self.w = nn.Linear(input_dim, hidden_dim, bias=False)
    self.v = nn.Linear(input_dim, hidden_dim, bias=False)
    self.output = nn.Linear(hidden_dim, input_dim, bias=False)
    self.silu = nn.SiLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.output(self.silu(self.w(x)) * self.v(x))


class _SparseTrackTaskEmbedder(nn.Module):
  """SparseTrack task embedder for full-reference task tokens."""

  def __init__(
    self,
    task_obs_dim: int,
    embedding_dim: int,
    reduced_task_dim: int | None = None,
    hidden_dims: tuple[int, ...] | list[int] | None = None,
  ) -> None:
    super().__init__()
    hidden_dims = tuple(hidden_dims or ())
    if reduced_task_dim is not None:
      self.task_projection = self._build_task_projection(
        task_obs_dim,
        reduced_task_dim,
        hidden_dims,
      )
      matrix = torch.randn(embedding_dim, reduced_task_dim, dtype=torch.float)
      q_matrix, r_matrix = torch.linalg.qr(matrix, mode="reduced")
      diag = torch.sign(torch.diag(r_matrix))
      diag[diag == 0] = 1.0
      self.register_buffer("W", q_matrix * diag)
      self._forward_method = self._reduced_task_projection
    else:
      self.task_projection = self._build_task_projection(
        task_obs_dim,
        embedding_dim,
        hidden_dims,
      )
      self._forward_method = self._normal_task_projection

  @staticmethod
  def _build_task_projection(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
  ) -> nn.Module:
    if len(hidden_dims) == 0:
      return nn.Linear(input_dim, output_dim)

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), nn.ELU()]
    for layer_index in range(len(hidden_dims) - 1):
      layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
      layers.append(nn.ELU())
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layers)

  def _reduced_task_projection(self, task_obs: torch.Tensor) -> torch.Tensor:
    task_embedding = self.task_projection(task_obs)
    task_embedding = task_embedding / (task_embedding.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.matmul(task_embedding, self.W.T)

  def _normal_task_projection(self, task_obs: torch.Tensor) -> torch.Tensor:
    return self.task_projection(task_obs)

  def forward(self, task_obs: torch.Tensor) -> torch.Tensor:
    return self._forward_method(task_obs)

  @torch.no_grad()
  def init_weights(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Linear):
        in_dim = module.weight.shape[1]
        std = 1.0 / math.sqrt(in_dim)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)


class _SparseTrackTransformerBlock(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int, ff_dim: int) -> None:
    super().__init__()
    self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    self.feed_forward = _SwiGLU(embed_dim, ff_dim)
    self.rmsnorm1 = _RMSNorm(embed_dim)
    self.rmsnorm2 = _RMSNorm(embed_dim)
    self.rmsnorm3 = _RMSNorm(embed_dim)
    self.cond_norm = _RMSNorm(embed_dim)
    self.rope = _RoPEPositionalEncoding(embed_dim)

  def forward(
    self,
    x: torch.Tensor,
    task_tokens: torch.Tensor,
    self_attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    x_norm = self.rmsnorm1(x)
    x_rope = self.rope(x_norm)
    attn_output, _ = self.self_attention(
      x_rope,
      x_rope,
      x_norm,
      attn_mask=self_attn_mask,
      need_weights=False,
    )
    x = x + attn_output

    x_norm2 = self.rmsnorm2(x)
    task_norm = self.cond_norm(task_tokens)
    cross_output, _ = self.cross_attention(
      x_norm2,
      task_norm,
      task_norm,
      need_weights=False,
    )
    x = x + cross_output

    return x + self.feed_forward(self.rmsnorm3(x))


class _BaseTrackingAttentionActor(MLPModel):
  """RSL-RL compatible base class for flattened-history tracking actors."""

  is_recurrent: ClassVar[bool] = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (1536, 1024, 512, 256),
    activation: str = "gelu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict | None = None,
    rnn_type: str | None = None,
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
    history_length: int = 11,
    frame_dim: int = 286,
    command_dim: int = 58,
    num_dofs: int = 29,
    d_model: int = 384,
    num_heads: int = 6,
    ffn_dim: int = 1536,
    history_layers: int = 0,
    cross_layers: int = 0,
    head_hidden_dims: tuple[int, ...] | list[int] = (1536, 1024, 512, 256),
    dropout: float = 0.0,
    attention_activation: str = "gelu",
    task_embedder_hidden_dims: tuple[int, ...] | list[int] | None = None,
    reduced_task_dim: int | None = None,
  ) -> None:
    self.history_length = int(history_length)
    self.frame_dim = int(frame_dim)
    self.command_dim = int(command_dim)
    self.num_dofs = int(num_dofs)
    self.d_model = int(d_model)
    self.num_heads = int(num_heads)
    self.ffn_dim = int(ffn_dim)
    self.history_layers = int(history_layers)
    self.cross_layers = int(cross_layers)
    self.dropout = float(dropout)
    self.attention_activation = attention_activation
    self.task_embedder_hidden_dims = tuple(task_embedder_hidden_dims or ())
    self.reduced_task_dim = reduced_task_dim
    self._latent_dim = self._attention_latent_dim()
    if cnn_cfg is not None:
      raise ValueError("tracking attention actors do not support cnn_cfg")
    if rnn_type is not None:
      raise ValueError("tracking attention actors do not support rnn_type")
    _ = (rnn_hidden_dim, rnn_num_layers)

    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=tuple(head_hidden_dims or hidden_dims),
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self._validate_config(output_dim)

  def _attention_latent_dim(self) -> int:
    raise NotImplementedError

  def _expected_output_dim(self) -> int:
    return self.num_dofs

  def _build_history_encoder(self, num_layers: int) -> nn.TransformerEncoder:
    encoder_layer = _make_encoder_layer(
      self.d_model,
      self.num_heads,
      self.ffn_dim,
      self.dropout,
      self.attention_activation,
    )
    return nn.TransformerEncoder(
      encoder_layer,
      num_layers=num_layers,
      norm=nn.LayerNorm(self.d_model),
      enable_nested_tensor=False,
    )

  def _build_cross_blocks(self, num_layers: int) -> nn.ModuleList:
    return nn.ModuleList(
      [
        _CrossAttentionBlock(
          self.d_model,
          self.num_heads,
          self.ffn_dim,
          self.dropout,
          self.attention_activation,
        )
        for _ in range(num_layers)
      ]
    )

  def _validate_config(self, output_dim: int) -> None:
    expected_flat_dim = self.history_length * self.frame_dim
    if self.obs_dim != expected_flat_dim:
      raise ValueError(
        f"expected flat observation dim {expected_flat_dim}, got {self.obs_dim}"
      )
    expected_frame_dim = sum(TERM_DIMS.values())
    if self.frame_dim != expected_frame_dim:
      raise ValueError(f"expected frame_dim {expected_frame_dim}, got {self.frame_dim}")
    if self.command_dim != 2 * self.num_dofs:
      raise ValueError(
        f"expected command_dim {2 * self.num_dofs}, got {self.command_dim}"
      )
    expected_output_dim = self._expected_output_dim()
    if output_dim != expected_output_dim:
      raise ValueError(f"expected output_dim {expected_output_dim}, got {output_dim}")
    if self.d_model % self.num_heads != 0:
      raise ValueError("d_model must be divisible by num_heads")

  def _get_latent_dim(self) -> int:
    return self._latent_dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    obs_list = [cast(torch.Tensor, obs[obs_group]) for obs_group in self.obs_groups]
    flat_obs = torch.cat(obs_list, dim=-1)
    flat_obs = self.obs_normalizer(flat_obs)
    return self._attention_latent_from_flat(flat_obs)

  def _term_history(self, flat_obs: torch.Tensor) -> dict[str, torch.Tensor]:
    terms: dict[str, torch.Tensor] = {}
    cursor = 0
    for name, dim in TERM_DIMS.items():
      next_cursor = cursor + self.history_length * dim
      terms[name] = flat_obs[:, cursor:next_cursor].reshape(
        flat_obs.shape[0],
        self.history_length,
        dim,
      )
      cursor = next_cursor
    return terms

  def _frame_history(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    return self._frame_history_from_terms(terms)

  def _proprio_history(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    return self._proprio_history_from_terms(terms)

  def _current_command_tokens(self, flat_obs: torch.Tensor) -> torch.Tensor:
    command = self._term_history(flat_obs)["command"][:, -1]
    return self._command_tokens(command)

  def _frame_history_from_terms(
    self,
    terms: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    return torch.cat([terms[name] for name in TERM_DIMS], dim=-1)

  def _proprio_history_from_terms(
    self,
    terms: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    return torch.cat([terms[name] for name in PROPRIO_TERMS], dim=-1)

  def _current_command_tokens_from_terms(
    self,
    terms: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    command = terms["command"][:, -1]
    return self._command_tokens(command)

  def _command_tokens(self, command: torch.Tensor) -> torch.Tensor:
    q_ref = command[:, : self.num_dofs]
    qd_ref = command[:, self.num_dofs :]
    return torch.stack((q_ref, qd_ref), dim=-1)

  def _attention_latent_from_flat(self, flat_obs: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  def _zero_mlp_input_weights_from(self, start_dim: int) -> None:
    first_layer = self.mlp[0]
    if not isinstance(first_layer, nn.Linear):
      raise TypeError("expected the MLP head to start with nn.Linear")
    with torch.no_grad():
      first_layer.weight[:, start_dim:] = 0.0

  def _init_mlp_input_weights_from(self, start_dim: int, std: float = 0.02) -> None:
    first_layer = self.mlp[0]
    if not isinstance(first_layer, nn.Linear):
      raise TypeError("expected the MLP head to start with nn.Linear")
    with torch.no_grad():
      nn.init.normal_(first_layer.weight[:, start_dim:], mean=0.0, std=std)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxTrackingAttentionActor(self, verbose)


class _OnnxTrackingAttentionActor(nn.Module):
  """ONNX wrapper that preserves custom attention preprocessing."""

  is_recurrent: bool = False

  def __init__(self, model: _BaseTrackingAttentionActor, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.model = copy.deepcopy(model)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.model.obs_normalizer(x)
    latent = self.model._attention_latent_from_flat(x)
    out = self.model.mlp(latent)
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class FullObsCausalAttentionActor(_BaseTrackingAttentionActor):
  """Humanoid-GPT-style causal attention over full observation frames."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.frame_proj = nn.Linear(self.frame_dim, self.d_model)
    self.pos_embedding = nn.Parameter(torch.empty(1, self.history_length, self.d_model))
    nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    self.history_encoder = self._build_history_encoder(self.history_layers)
    self.register_buffer(
      "_causal_mask",
      _make_causal_mask(self.history_length),
      persistent=False,
    )
    self._init_mlp_input_weights_from(self.frame_dim)

  def _attention_latent_dim(self) -> int:
    return self.frame_dim + self.d_model

  def _attention_latent_from_flat(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    frames = self._frame_history_from_terms(terms)
    tokens = self.frame_proj(frames) + self.pos_embedding
    encoded = self.history_encoder(tokens, mask=self._causal_mask)
    return torch.cat((frames[:, -1], encoded[:, -1]), dim=-1)


class ProprioRefCrossAttentionActor(_BaseTrackingAttentionActor):
  """Cross-attention from proprioceptive history summary to current ref tokens."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.proprio_proj = nn.Linear(93, self.d_model)
    self.proprio_pos_embedding = nn.Parameter(
      torch.empty(1, self.history_length, self.d_model)
    )
    nn.init.trunc_normal_(self.proprio_pos_embedding, std=0.02)
    self.command_token_proj = nn.Linear(2, self.d_model)
    self.joint_embedding = nn.Parameter(torch.zeros(1, self.num_dofs, self.d_model))
    self.history_pool = nn.Linear(self.history_length * self.d_model, self.d_model)
    self._init_current_token_history_pool()
    self.query_norm = nn.LayerNorm(self.d_model)
    self.cross_blocks = self._build_cross_blocks(self.cross_layers)

  def _attention_latent_dim(self) -> int:
    return self.d_model

  def _attention_latent_from_flat(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    proprio = self._proprio_history_from_terms(terms)
    proprio_tokens = self.proprio_proj(proprio) + self.proprio_pos_embedding
    query = self.history_pool(proprio_tokens.flatten(start_dim=1)).unsqueeze(1)
    query = self.query_norm(query)

    ref_tokens = self.command_token_proj(self._current_command_tokens_from_terms(terms))
    ref_tokens = ref_tokens + self.joint_embedding

    for block in self.cross_blocks:
      query = block(query, ref_tokens)
    return query.squeeze(1)

  def _init_current_token_history_pool(self) -> None:
    nn.init.zeros_(self.history_pool.weight)
    nn.init.zeros_(self.history_pool.bias)
    current_offset = (self.history_length - 1) * self.d_model
    with torch.no_grad():
      for dim_idx in range(self.d_model):
        self.history_pool.weight[dim_idx, current_offset + dim_idx] = 1.0


class HistProprioCrossAttentionActor(_BaseTrackingAttentionActor):
  """RoHM-style causal proprio history encoder plus command cross-attention."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.proprio_proj = nn.Linear(93, self.d_model)
    self.proprio_pos_embedding = nn.Parameter(
      torch.empty(1, self.history_length, self.d_model)
    )
    nn.init.trunc_normal_(self.proprio_pos_embedding, std=0.02)
    self.history_encoder = self._build_history_encoder(self.history_layers)
    self.command_token_proj = nn.Linear(2, self.d_model)
    self.joint_embedding = nn.Parameter(torch.zeros(1, self.num_dofs, self.d_model))
    self.cross_blocks = self._build_cross_blocks(self.cross_layers)
    self.register_buffer(
      "_causal_mask",
      _make_causal_mask(self.history_length),
      persistent=False,
    )
    self._init_mlp_input_weights_from(self.frame_dim)

  def _attention_latent_dim(self) -> int:
    return self.frame_dim + 2 * self.d_model

  def _attention_latent_from_flat(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    proprio = self._proprio_history_from_terms(terms)
    proprio_tokens = self.proprio_proj(proprio) + self.proprio_pos_embedding
    dynamics_tokens = self.history_encoder(proprio_tokens, mask=self._causal_mask)
    dynamics = dynamics_tokens[:, -1]

    ref_tokens = self.command_token_proj(self._current_command_tokens_from_terms(terms))
    ref_tokens = ref_tokens + self.joint_embedding
    command_query = dynamics.unsqueeze(1)
    for block in self.cross_blocks:
      command_query = block(command_query, ref_tokens)
    command_embedding = command_query.squeeze(1)

    current_full_obs = self._frame_history_from_terms(terms)[:, -1]
    return torch.cat((current_full_obs, dynamics, command_embedding), dim=-1)


class HistProprioCrossAttentionCritic(HistProprioCrossAttentionActor):
  """RoHM-style history/cross-attention critic for tracking value estimation."""

  def _expected_output_dim(self) -> int:
    return 1

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    raise NotImplementedError(
      "HistProprioCross transformer critic is not exported to ONNX"
    )


class _SparseTrackFullRefAttentionMixin:
  """Shared SparseTrack full-reference transformer implementation."""

  def _init_sparsetrack_modules(self, output_dim: int) -> None:
    self.mlp = nn.Sequential(nn.Linear(self.d_model, output_dim))
    self.prop_obs_dim = sum(TERM_DIMS[name] for name in SPARSETRACK_PROP_TERMS)
    self.task_obs_dim = sum(TERM_DIMS[name] for name in SPARSETRACK_TASK_TERMS)

    self.prop_projection = nn.Linear(self.prop_obs_dim, self.d_model)
    self.action_projection = nn.Linear(self.num_dofs, self.d_model)
    self.task_embedder = _SparseTrackTaskEmbedder(
      task_obs_dim=self.task_obs_dim,
      embedding_dim=self.d_model,
      reduced_task_dim=self.reduced_task_dim,
      hidden_dims=self.task_embedder_hidden_dims,
    )
    self.empty_embedding = nn.Parameter(torch.empty(1, 1, self.d_model))
    self.transformer_blocks = nn.ModuleList(
      [
        _SparseTrackTransformerBlock(
          embed_dim=self.d_model,
          num_heads=self.num_heads,
          ff_dim=self.ffn_dim,
        )
        for _ in range(self.history_layers)
      ]
    )
    self.final_norm = _RMSNorm(self.d_model)
    self.register_buffer(
      "_empty_token_mask",
      self._make_empty_token_mask(self.history_length),
      persistent=False,
    )
    self._init_sparsetrack_weights()
    self._zero_output_head()

  def _attention_latent_dim(self) -> int:
    return self.d_model

  @staticmethod
  def _make_empty_token_mask(history_length: int) -> torch.Tensor:
    seq_len = 2 * history_length
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    mask[:-1, -1] = True
    return mask

  def _attention_latent_from_flat(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    prop_obs = self._sparsetrack_prop_history_from_terms(terms)
    action_obs = terms["actions"]
    task_obs = self._sparsetrack_task_history_from_terms(terms)

    prop_tokens = self.prop_projection(prop_obs)
    action_tokens = self.action_projection(action_obs)
    task_tokens = self.task_embedder(task_obs)

    context = prop_tokens.new_empty(
      prop_tokens.shape[0],
      2 * self.history_length - 1,
      self.d_model,
    )
    context[:, 0::2] = prop_tokens
    context[:, 1::2] = action_tokens[:, 1:]

    empty_token = self.empty_embedding.expand(prop_tokens.shape[0], -1, -1)
    x = torch.cat((context, empty_token), dim=1)
    mask = self._empty_token_mask.to(device=x.device)

    for block in self.transformer_blocks:
      x = block(x, task_tokens, self_attn_mask=mask)
    x = self.final_norm(x)
    return x[:, -1]

  def _sparsetrack_prop_history_from_terms(
    self,
    terms: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    return torch.cat([terms[name] for name in SPARSETRACK_PROP_TERMS], dim=-1)

  def _sparsetrack_task_history_from_terms(
    self,
    terms: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    return torch.cat([terms[name] for name in SPARSETRACK_TASK_TERMS], dim=-1)

  def _init_sparsetrack_weights(self) -> None:
    num_layers = max(len(self.transformer_blocks), 1)
    res_scale = 1.0 / math.sqrt(2.0 * num_layers)

    for module in (
      self.prop_projection,
      self.action_projection,
    ):
      self._init_linear(module)
    self.task_embedder.init_weights()

    for block in self.transformer_blocks:
      for module in block.modules():
        if isinstance(module, nn.MultiheadAttention):
          continue
        if isinstance(module, _RMSNorm):
          nn.init.ones_(module.scale)
        elif isinstance(module, nn.Linear):
          self._init_linear(module)

      for module in block.modules():
        if isinstance(module, nn.MultiheadAttention):
          embed_dim = module.embed_dim
          std = 1.0 / math.sqrt(embed_dim)
          nn.init.normal_(module.in_proj_weight, mean=0.0, std=std)
          if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
          out_std = std / math.sqrt(2.0 * num_layers)
          nn.init.normal_(module.out_proj.weight, mean=0.0, std=out_std)
          if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)

      with torch.no_grad():
        block.feed_forward.output.weight.mul_(res_scale)

    nn.init.ones_(self.final_norm.scale)
    nn.init.trunc_normal_(self.empty_embedding, std=0.02)

  @staticmethod
  def _init_linear(module: nn.Linear) -> None:
    in_dim = module.weight.shape[1]
    std = 1.0 / math.sqrt(in_dim)
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
      nn.init.zeros_(module.bias)

  def _zero_output_head(self) -> None:
    last_linear = next(
      (module for module in reversed(self.mlp) if isinstance(module, nn.Linear)),
      None,
    )
    if last_linear is None:
      raise TypeError("expected the MLP head to contain an nn.Linear output layer")
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)


class SparseTrackFullRefAttentionActor(
  _SparseTrackFullRefAttentionMixin,
  _BaseTrackingAttentionActor,
):
  """SparseTrack-style transformer actor over full-ref tracking observations."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._init_sparsetrack_modules(self.num_dofs)


class SparseTrackFullRefAttentionCritic(
  _SparseTrackFullRefAttentionMixin,
  _BaseTrackingAttentionActor,
):
  """SparseTrack-style transformer critic over full-ref tracking observations."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._init_sparsetrack_modules(1)

  def _expected_output_dim(self) -> int:
    return 1

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    raise NotImplementedError("SparseTrack transformer critic is not exported to ONNX")
