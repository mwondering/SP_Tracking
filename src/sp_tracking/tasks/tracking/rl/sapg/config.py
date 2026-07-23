"""Configuration for the SAPG/CPO ensemble-policy extension."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SAPGConfig:
  """Shared ensemble options plus the official CPO follower objectives."""

  enabled: bool = False
  method: str = "sapg"
  compatibility: str = "official"
  num_policy_blocks: int = 4
  local_parameter_dim: int = 32
  off_policy_ratio: int = 1
  exploration_type: str = "none"
  entropy_coef_scale: float = 1.0
  value_eval_chunk_size: int = 8192
  cpo_awac_temperature: float = 0.2
  cpo_awac_max_weight: float = 100.0
  cpo_awac_coef: float = 0.001
  cpo_kl_coef: float = 0.0

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
      raise TypeError("Ensemble-policy enabled must be a boolean")
    if not isinstance(self.method, str):
      raise TypeError("Ensemble-policy method must be a string")
    if not isinstance(self.compatibility, str):
      raise TypeError("Ensemble-policy compatibility must be a string")
    if not isinstance(self.exploration_type, str):
      raise TypeError("Ensemble-policy exploration_type must be a string")
    for name in (
      "num_policy_blocks",
      "local_parameter_dim",
      "off_policy_ratio",
      "value_eval_chunk_size",
    ):
      if type(getattr(self, name)) is not int:
        raise TypeError(f"Ensemble-policy {name} must be an integer")
    if (
      isinstance(self.entropy_coef_scale, bool)
      or not isinstance(self.entropy_coef_scale, (int, float))
      or not math.isfinite(float(self.entropy_coef_scale))
    ):
      raise TypeError(
        "Ensemble-policy entropy_coef_scale must be a finite number"
      )
    for name in (
      "cpo_awac_temperature",
      "cpo_awac_max_weight",
      "cpo_awac_coef",
      "cpo_kl_coef",
    ):
      value = getattr(self, name)
      if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
      ):
        raise TypeError(f"CPO {name} must be a finite number")
    if self.method not in {"sapg", "cpo"}:
      raise ValueError("Ensemble-policy method must be 'sapg' or 'cpo'")
    if self.compatibility != "official":
      raise ValueError(
        "Ensemble-policy compatibility currently supports only 'official'"
      )
    if self.num_policy_blocks < 2:
      raise ValueError("Ensemble-policy num_policy_blocks must be at least 2")
    if self.local_parameter_dim <= 0:
      raise ValueError("Ensemble-policy local_parameter_dim must be positive")
    if not 1 <= self.off_policy_ratio < self.num_policy_blocks:
      raise ValueError(
        "Ensemble-policy off_policy_ratio must be in "
        "[1, num_policy_blocks - 1]"
      )
    if self.exploration_type not in {"none", "entropy"}:
      raise ValueError(
        "Ensemble-policy exploration_type must be 'none' or 'entropy'"
      )
    if self.entropy_coef_scale < 0.0:
      raise ValueError(
        "Ensemble-policy entropy_coef_scale must be non-negative"
      )
    if self.value_eval_chunk_size <= 0:
      raise ValueError(
        "Ensemble-policy value_eval_chunk_size must be positive"
      )
    if self.cpo_awac_temperature <= 0.0:
      raise ValueError("CPO cpo_awac_temperature must be positive")
    if self.cpo_awac_max_weight <= 0.0:
      raise ValueError("CPO cpo_awac_max_weight must be positive")
    if self.cpo_awac_coef < 0.0:
      raise ValueError("CPO cpo_awac_coef must be non-negative")
    if self.cpo_kl_coef < 0.0:
      raise ValueError("CPO cpo_kl_coef must be non-negative")

  @property
  def leader_policy_id(self) -> int:
    return self.num_policy_blocks - 1

  @property
  def is_cpo(self) -> bool:
    return self.method == "cpo"

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)
