"""Configuration and validation for the official-compatible SAPG extension."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SAPGConfig:
  """Options whose semantics follow the official SAPG implementation."""

  enabled: bool = False
  compatibility: str = "official"
  num_policy_blocks: int = 4
  local_parameter_dim: int = 32
  off_policy_ratio: int = 1
  exploration_type: str = "none"
  entropy_coef_scale: float = 1.0
  value_eval_chunk_size: int = 8192

  def __post_init__(self) -> None:
    self.validate()

  @classmethod
  def from_dict(cls, value: dict[str, Any] | None) -> SAPGConfig:
    if value is None:
      return cls()
    if not isinstance(value, dict):
      raise TypeError("algorithm.sapg_cfg must be a mapping")
    unknown = set(value).difference(cls.__dataclass_fields__)
    if unknown:
      names = ", ".join(sorted(unknown))
      raise ValueError(f"Unknown algorithm.sapg_cfg option(s): {names}")
    normalized = dict(value)
    for name in (
      "num_policy_blocks",
      "local_parameter_dim",
      "off_policy_ratio",
      "value_eval_chunk_size",
    ):
      raw = normalized.get(name)
      if isinstance(raw, float) and math.isfinite(raw) and raw.is_integer():
        normalized[name] = int(raw)
    return cls(**normalized)

  def validate(self) -> None:
    if type(self.enabled) is not bool:
      raise TypeError("SAPG enabled must be a boolean")
    if not isinstance(self.compatibility, str):
      raise TypeError("SAPG compatibility must be a string")
    if not isinstance(self.exploration_type, str):
      raise TypeError("SAPG exploration_type must be a string")
    for name in (
      "num_policy_blocks",
      "local_parameter_dim",
      "off_policy_ratio",
      "value_eval_chunk_size",
    ):
      if type(getattr(self, name)) is not int:
        raise TypeError(f"SAPG {name} must be an integer")
    if (
      isinstance(self.entropy_coef_scale, bool)
      or not isinstance(self.entropy_coef_scale, (int, float))
      or not math.isfinite(float(self.entropy_coef_scale))
    ):
      raise TypeError("SAPG entropy_coef_scale must be a finite number")
    if self.compatibility != "official":
      raise ValueError("SAPG compatibility currently supports only 'official'")
    if self.num_policy_blocks < 2:
      raise ValueError("SAPG num_policy_blocks must be at least 2")
    if self.local_parameter_dim <= 0:
      raise ValueError("SAPG local_parameter_dim must be positive")
    if not 1 <= self.off_policy_ratio < self.num_policy_blocks:
      raise ValueError(
        "SAPG off_policy_ratio must be in [1, num_policy_blocks - 1]"
      )
    if self.exploration_type not in {"none", "entropy"}:
      raise ValueError("SAPG exploration_type must be 'none' or 'entropy'")
    if self.entropy_coef_scale < 0.0:
      raise ValueError("SAPG entropy_coef_scale must be non-negative")
    if self.value_eval_chunk_size <= 0:
      raise ValueError("SAPG value_eval_chunk_size must be positive")

  @property
  def leader_policy_id(self) -> int:
    return self.num_policy_blocks - 1

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)
