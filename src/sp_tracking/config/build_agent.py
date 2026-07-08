from __future__ import annotations

from dataclasses import fields
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _to_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
  return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)


def _filter_dataclass_kwargs(cls, data: dict[str, Any]) -> dict[str, Any]:
  names = {field.name for field in fields(cls)}
  result = {key: value for key, value in data.items() if key in names}
  if "hidden_dims" in result and isinstance(result["hidden_dims"], list):
    result["hidden_dims"] = tuple(result["hidden_dims"])
  if "wandb_tags" in result and isinstance(result["wandb_tags"], list):
    result["wandb_tags"] = tuple(result["wandb_tags"])
  return result


def build_agent_cfg(cfg: DictConfig | dict[str, Any]) -> RslRlOnPolicyRunnerCfg:
  data = _to_container(cfg)
  actor = RslRlModelCfg(**_filter_dataclass_kwargs(RslRlModelCfg, dict(data.pop("actor"))))
  critic = RslRlModelCfg(**_filter_dataclass_kwargs(RslRlModelCfg, dict(data.pop("critic"))))
  algorithm = RslRlPpoAlgorithmCfg(
    **_filter_dataclass_kwargs(RslRlPpoAlgorithmCfg, dict(data.pop("algorithm")))
  )
  runner_kwargs = _filter_dataclass_kwargs(RslRlOnPolicyRunnerCfg, data)
  return RslRlOnPolicyRunnerCfg(
    actor=actor,
    critic=critic,
    algorithm=algorithm,
    **runner_kwargs,
  )
