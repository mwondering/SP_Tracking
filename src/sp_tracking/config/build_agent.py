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


@dataclass
class HeftTeacherPpoAlgorithmCfg(SplitLrPpoAlgorithmCfg):
  """Teacher-only HEFT pretrain options used exclusively by the SP agent."""

  entropy_coef_start: float = 0.01
  entropy_coef_end: float = 0.005
  desired_kl_upper: tuple[tuple[float, float], ...] = (
    (0.0, 0.015),
    (0.15, 0.015),
    (0.2, 0.01),
    (0.8, 0.0075),
    (1.0, 0.0075),
  )
  lr_schedule_scale_factor: float = 1.05
  lr_schedule_min: float = 1.0e-7
  lr_schedule_max: float = 1.0e-3
  symmetry_cfg: dict[str, Any] | None = None


@dataclass
class HeftTeacherActorCfg(RslRlModelCfg):
  privileged_latent_dim: int = 256
  vecnorm_decay: float = 0.9999
  init_std: tuple[float, ...] | None = None


@dataclass
class HeftTeacherCriticCfg(RslRlModelCfg):
  vecnorm_decay: float = 0.9999


@dataclass
class SPV3EstimatorActorCfg(RslRlModelCfg):
  estimator_hidden_dims: tuple[int, ...] = (512, 256, 128)
  estimator_activation: str = "elu"
  actor_core_group: str = "actor_core"
  estimator_history_group: str = "estimator_history"
  estimator_target_group: str = "estimator_target"
  estimator_history_length: int = 50
  policy_history_length: int = 5


@dataclass
class SPV4KeyBodyActorCfg(SPV3EstimatorActorCfg):
  robot_key_body_group: str = "robot_key_body"
  ref_key_body_group: str = "ref_key_body"
  key_body_error_group: str = "key_body_error"


@dataclass
class SPV3EstimatorPpoAlgorithmCfg(SplitLrPpoAlgorithmCfg):
  estimator_root_height_loss_coef: float = 1.0
  estimator_root_lin_vel_loss_coef: float = 1.0


def _to_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
  return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)


def _filter_dataclass_kwargs(cls, data: dict[str, Any]) -> dict[str, Any]:
  names = {field.name for field in fields(cls)}
  result = {key: value for key, value in data.items() if key in names}
  for key, value in tuple(result.items()):
    if key.endswith("hidden_dims") and isinstance(value, list):
      result[key] = tuple(value)
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
  actor_data = dict(data.pop("actor"))
  critic_data = dict(data.pop("critic"))
  actor_class_name = str(actor_data.get("class_name", ""))
  if actor_class_name.endswith(":HeftTeacherActor"):
    actor_cls = HeftTeacherActorCfg
  elif actor_class_name.endswith(":SPV3EstimatorActor"):
    actor_cls = SPV3EstimatorActorCfg
  elif actor_class_name.endswith(":SPV4KeyBodyActor"):
    actor_cls = SPV4KeyBodyActorCfg
  else:
    actor_cls = RslRlModelCfg
  critic_cls = (
    HeftTeacherCriticCfg
    if str(critic_data.get("class_name", "")).endswith(":HeftTeacherCritic")
    else RslRlModelCfg
  )
  actor = actor_cls(**_filter_dataclass_kwargs(actor_cls, actor_data))
  critic = critic_cls(**_filter_dataclass_kwargs(critic_cls, critic_data))
  algorithm_data = dict(data.pop("algorithm"))
  split_lr_keys = {"actor_learning_rate", "critic_learning_rate", "clamp_rewards_min"}
  algorithm_class_name = str(algorithm_data.get("class_name", ""))
  if algorithm_class_name.endswith(":HeftTeacherPPO"):
    algorithm_cls = HeftTeacherPpoAlgorithmCfg
  elif algorithm_class_name.endswith(":SPV3EstimatorPPO"):
    algorithm_cls = SPV3EstimatorPpoAlgorithmCfg
  elif split_lr_keys.intersection(algorithm_data):
    algorithm_cls = SplitLrPpoAlgorithmCfg
  else:
    algorithm_cls = RslRlPpoAlgorithmCfg
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
    if model.get("class_name") == "MLPModel" or str(
      model.get("class_name", "")
    ).endswith(
      (
        ":HeftTeacherActor",
        ":HeftTeacherCritic",
        ":SPV3EstimatorActor",
        ":SPV4KeyBodyActor",
      )
    ):
      for key in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        model.pop(key, None)
  return data
