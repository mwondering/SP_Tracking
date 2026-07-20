from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class SapgPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Optional SAPG extension shared by plain and tracking-specific PPOs."""

  sapg_cfg: dict[str, Any] | None = None


@dataclass
class SplitLrPpoAlgorithmCfg(SapgPpoAlgorithmCfg):
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
class SPV5ReferenceEncoderActorCfg(SPV3EstimatorActorCfg):
  reference_encoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  reference_encoder_activation: str = "elu"
  robot_root_quat_group: str = "robot_root_quat"
  reference_encoder_input_group: str = "reference_encoder_input"
  reference_encoder_target_group: str = "reference_encoder_target"
  robot_key_body_group: str = "robot_key_body"
  reference_fps: float = 50.0
  keypoint_specs: tuple[dict[str, Any], ...] = ()


@dataclass
class SPV51ContactEstimatorActorCfg(SPV5ReferenceEncoderActorCfg):
  foot_contact_target_group: str = "foot_contact_target"


@dataclass
class SPV51ContactEstimatorMoEActorCfg(SPV51ContactEstimatorActorCfg):
  moe_context_hidden_dim: int = 1285
  moe_hidden_dim: int = 256
  moe_num_experts: int = 16
  moe_top_k: int = 8
  moe_expansion: int = 4
  moe_router_temperature: float = 1.0
  moe_router_init_std: float = 1.0e-2


@dataclass
class SPV6RmaActorCfg(SPV5ReferenceEncoderActorCfg):
  rma_physics_nominal_group: str = "rma_physics_nominal"
  rma_global_latent_dim: int = 8
  rma_sensor_latent_dim: int = 32
  rma_push_latent_dim: int = 16


@dataclass
class SPV61DirectActorCfg(SPV5ReferenceEncoderActorCfg):
  rma_physics_actual_group: str = "rma_physics_actual"
  rma_push_history_group: str = "rma_push_history"


@dataclass
class SPV6RmaCriticCfg(HeftTeacherCriticCfg):
  rma_physics_actual_group: str = "rma_physics_actual"
  rma_push_history_group: str = "rma_push_history"
  rma_global_latent_dim: int = 8
  rma_sensor_latent_dim: int = 32
  rma_push_latent_dim: int = 16


@dataclass
class SPV61DirectCriticCfg(HeftTeacherCriticCfg):
  rma_physics_actual_group: str = "rma_physics_actual"
  rma_push_history_group: str = "rma_push_history"


@dataclass
class SPV3EstimatorPpoAlgorithmCfg(SplitLrPpoAlgorithmCfg):
  estimator_learning_rate: float = 1.0e-4
  estimator_root_height_loss_coef: float = 1.0
  estimator_root_lin_vel_loss_coef: float = 1.0


@dataclass
class SPV5ReferenceEncoderPpoAlgorithmCfg(SPV3EstimatorPpoAlgorithmCfg):
  reference_encoder_loss_coef: float = 1.0


@dataclass
class PolicyGradientDiagnosticsPpoAlgorithmCfg(
  SPV5ReferenceEncoderPpoAlgorithmCfg
):
  gradient_motion_label_group: str = "gradient_motion_label"
  gradient_motion_phase_group: str = "gradient_motion_phase"
  gradient_stratified_minibatches: bool = True
  gradient_diagnostics_eps: float = 1.0e-12


@dataclass
class SPV51ContactEstimatorPpoAlgorithmCfg(
  SPV5ReferenceEncoderPpoAlgorithmCfg
):
  estimator_foot_contact_loss_coef: float = 0.1


@dataclass
class SPV51ContactEstimatorMoEPpoAlgorithmCfg(
  SPV51ContactEstimatorPpoAlgorithmCfg
):
  moe_balance_loss_coef: float = 3.0e-3
  moe_confidence_loss_coef: float = 3.0e-4
  moe_confidence_warmup_updates: int = 5000
  moe_confidence_ramp_updates: int = 10000
  moe_collect_chunk_size: int = 4096


@dataclass
class SPV6RmaPpoAlgorithmCfg(SPV5ReferenceEncoderPpoAlgorithmCfg):
  rma_global_alignment_coef: float = 1.0
  rma_sensor_alignment_coef: float = 0.5
  rma_push_alignment_coef: float = 1.0
  rma_physics_reconstruction_coef: float = 0.1
  rma_push_reconstruction_coef: float = 0.1


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
    # Resolve task-relative interpolations while ``overrides`` is still
    # attached to the composed Hydra root.  Detaching it first would make
    # values such as ${task.obs.semantic_keypoints.heft} unresolvable.
    resolved_overrides = _to_container(overrides)
    merged = OmegaConf.merge(OmegaConf.create(data), resolved_overrides)
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
  elif actor_class_name.endswith(":SPV51ContactEstimatorMoEActor"):
    actor_cls = SPV51ContactEstimatorMoEActorCfg
  elif actor_class_name.endswith(":SPV51ContactEstimatorActor"):
    actor_cls = SPV51ContactEstimatorActorCfg
  elif actor_class_name.endswith(":SPV5ReferenceEncoderActor"):
    actor_cls = SPV5ReferenceEncoderActorCfg
  elif actor_class_name.endswith(":SPV6RmaActor"):
    actor_cls = SPV6RmaActorCfg
  elif actor_class_name.endswith(":SPV61DirectActor"):
    actor_cls = SPV61DirectActorCfg
  else:
    actor_cls = RslRlModelCfg
  critic_class_name = str(critic_data.get("class_name", ""))
  if critic_class_name.endswith(":SPV6RmaCritic"):
    critic_cls = SPV6RmaCriticCfg
  elif critic_class_name.endswith(":SPV61DirectCritic"):
    critic_cls = SPV61DirectCriticCfg
  elif critic_class_name.endswith(":HeftTeacherCritic"):
    critic_cls = HeftTeacherCriticCfg
  else:
    critic_cls = RslRlModelCfg
  actor = actor_cls(**_filter_dataclass_kwargs(actor_cls, actor_data))
  critic = critic_cls(**_filter_dataclass_kwargs(critic_cls, critic_data))
  algorithm_data = dict(data.pop("algorithm"))
  split_lr_keys = {"actor_learning_rate", "critic_learning_rate", "clamp_rewards_min"}
  algorithm_class_name = str(algorithm_data.get("class_name", ""))
  sapg_data = algorithm_data.get("sapg_cfg")
  sapg_enabled = isinstance(sapg_data, dict) and bool(sapg_data.get("enabled", False))
  if sapg_enabled and algorithm_class_name == "PPO":
    algorithm_class_name = (
      "sp_tracking.tasks.tracking.rl.ppo:SparseTrackSplitLrPPO"
    )
    algorithm_data["class_name"] = algorithm_class_name
  if algorithm_class_name.endswith(":HeftTeacherPPO"):
    algorithm_cls = HeftTeacherPpoAlgorithmCfg
  elif algorithm_class_name.endswith(":PolicyGradientDiagnosticsPPO"):
    algorithm_cls = PolicyGradientDiagnosticsPpoAlgorithmCfg
  elif algorithm_class_name.endswith(":SPV6RmaPPO"):
    algorithm_cls = SPV6RmaPpoAlgorithmCfg
  elif algorithm_class_name.endswith(":SPV51ContactEstimatorMoEPPO"):
    algorithm_cls = SPV51ContactEstimatorMoEPpoAlgorithmCfg
  elif algorithm_class_name.endswith(":SPV51ContactEstimatorPPO"):
    algorithm_cls = SPV51ContactEstimatorPpoAlgorithmCfg
  elif algorithm_class_name.endswith(":SPV3EstimatorPPO"):
    algorithm_cls = SPV3EstimatorPpoAlgorithmCfg
  elif algorithm_class_name.endswith(":SPV5ReferenceEncoderPPO"):
    algorithm_cls = SPV5ReferenceEncoderPpoAlgorithmCfg
  elif split_lr_keys.intersection(algorithm_data):
    algorithm_cls = SplitLrPpoAlgorithmCfg
  elif sapg_enabled:
    algorithm_cls = SapgPpoAlgorithmCfg
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
  sapg_cfg = data["algorithm"].get("sapg_cfg")
  if not isinstance(sapg_cfg, dict) or not bool(sapg_cfg.get("enabled", False)):
    # This is the hard non-regression boundary: disabled SAPG never reaches
    # RSL-RL's PPO constructor and cannot alter its model or update path.
    data["algorithm"].pop("sapg_cfg", None)
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
        ":SPV5ReferenceEncoderActor",
        ":SPV51ContactEstimatorActor",
        ":SPV51ContactEstimatorMoEActor",
        ":SPV6RmaActor",
        ":SPV6RmaCritic",
        ":SPV61DirectActor",
        ":SPV61DirectCritic",
      )
    ):
      for key in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        model.pop(key, None)
  return data
