from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class SplitLrPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO options implemented by the tracking-specific PPO subclass."""

  actor_learning_rate: float | None = None
  critic_learning_rate: float | None = None
  # ``None`` preserves normal RSL-RL handling.  SP enables 0.0 to match the
  # reference PPO's non-negative reward signal before GAE.
  clamp_rewards_min: float | None = None


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


def build_agent_cfg(
  cfg: DictConfig | dict[str, Any],
  overrides: DictConfig | dict[str, Any] | None = None,
) -> RslRlOnPolicyRunnerCfg:
  data = _to_container(cfg)
  if overrides:
    merged = OmegaConf.merge(OmegaConf.create(data), overrides)
    data = OmegaConf.to_container(merged, resolve=True)
    assert isinstance(data, dict)
  actor = RslRlModelCfg(**_filter_dataclass_kwargs(RslRlModelCfg, dict(data.pop("actor"))))
  critic = RslRlModelCfg(**_filter_dataclass_kwargs(RslRlModelCfg, dict(data.pop("critic"))))
  algorithm_data = dict(data.pop("algorithm"))
  split_lr_keys = {"actor_learning_rate", "critic_learning_rate", "clamp_rewards_min"}
  algorithm_cls = (
    SplitLrPpoAlgorithmCfg
    if split_lr_keys.intersection(algorithm_data)
    else RslRlPpoAlgorithmCfg
  )
  algorithm = algorithm_cls(
    **_filter_dataclass_kwargs(algorithm_cls, algorithm_data)
  )
  runner_kwargs = _filter_dataclass_kwargs(RslRlOnPolicyRunnerCfg, data)
  if "obs_groups" in runner_kwargs:
    runner_kwargs["obs_groups"] = {
      str(name): tuple(groups)
      for name, groups in runner_kwargs["obs_groups"].items()
    }
  return RslRlOnPolicyRunnerCfg(
    actor=actor,
    critic=critic,
    algorithm=algorithm,
    **runner_kwargs,
  )


def serialize_agent_cfg(cfg: RslRlOnPolicyRunnerCfg) -> dict[str, Any]:
  """Convert mjlab's broad dataclass config to constructor-ready RSL config.

  ``RslRlModelCfg`` exposes CNN/RNN options for several model classes, while
  RSL-RL's ``MLPModel`` does not accept those unused dataclass defaults as
  keyword arguments.  Keeping this conversion here makes every task profile
  (including the SP profile) runnable rather than merely composable.
  """
  data = asdict(cfg)
  for model_name in ("actor", "critic"):
    model = data[model_name]
    if model.get("class_name") == "MLPModel":
      for key in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        model.pop(key, None)
  return data
