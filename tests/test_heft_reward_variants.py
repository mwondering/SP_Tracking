from __future__ import annotations

from dataclasses import fields
from inspect import Parameter, signature

from hydra import compose, initialize_config_module
from mjlab.tasks.registry import list_tasks

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.task_catalog import (
  TASK_BY_CONFIG_NAME,
  TASK_SPECS,
)


PARENT_BY_HEFT_REWARD = {
  "tracking_bfm_heft_reward": "tracking_bfm",
  "tracking_bfm_sp_ablation_bfm_actor_heft_reward": (
    "tracking_bfm_sp_ablation_bfm_actor"
  ),
  "tracking_bfm_sp_ablation_student_actor_heft_reward": (
    "tracking_bfm_sp_ablation_student_actor"
  ),
  "tracking_bfm_sp_ablation_teacher_actor_heft_reward": (
    "tracking_bfm_sp_ablation_teacher_actor"
  ),
  "tracking_bfm_student_actor_bfm_critic_heft_reward": (
    "tracking_bfm_student_actor_bfm_critic"
  ),
  "tracking_bfm_teacher_actor_bfm_critic_heft_reward": (
    "tracking_bfm_teacher_actor_bfm_critic"
  ),
  "tracking_bfm_wbteleop_actor_bfm_critic_heft_reward": (
    "tracking_bfm_wbteleop_actor_bfm_critic"
  ),
  "tracking_bfm_wbteleop_actor_heft_critic_heft_reward": (
    "tracking_bfm_wbteleop_actor_heft_critic"
  ),
}


def _compose(task_name: str):
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=[f"task={task_name}"])


def test_every_non_sp_task_has_a_registered_heft_reward_variant() -> None:
  # Import explicitly because importing build_env can make mjlab's eager
  # entry-point discovery hit the registry during a temporary circular import.
  import sp_tracking.tasks.tracking.registry  # noqa: F401

  original_names = {
    spec.config_name
    for spec in TASK_SPECS
    if not spec.config_name.endswith("_heft_reward")
  }
  assert original_names == {
    "tracking_bfm",
    "tracking_bfm_sp",
    "tracking_bfm_sp_ablation_bfm_actor",
    "tracking_bfm_sp_ablation_student_actor",
    "tracking_bfm_sp_ablation_teacher_actor",
    "tracking_bfm_student_actor_bfm_critic",
    "tracking_bfm_teacher_actor_bfm_critic",
    "tracking_bfm_wbteleop_actor_bfm_critic",
    "tracking_bfm_wbteleop_actor_heft_critic",
  }
  assert set(PARENT_BY_HEFT_REWARD.values()) == original_names - {
    "tracking_bfm_sp"
  }
  assert set(PARENT_BY_HEFT_REWARD) <= set(TASK_BY_CONFIG_NAME)
  registered = set(list_tasks())
  assert {
    TASK_BY_CONFIG_NAME[name].task_id for name in PARENT_BY_HEFT_REWARD
  } <= registered


def test_heft_reward_variants_preserve_parent_policy_and_runtime_semantics() -> None:
  allowed_command_differences = {
    "body_names",
    "feet_standing_body_names",
    "feet_standing",
  }
  for variant_name, parent_name in PARENT_BY_HEFT_REWARD.items():
    parent = prepare_train_cfg(_compose(parent_name))
    variant = prepare_train_cfg(_compose(variant_name))

    assert variant.raw.task.name == variant_name
    assert variant.raw.task.variant.reward_profile == "heft_tracking_bfm"
    assert variant.agent == parent.agent
    assert variant.env.observations == parent.env.observations
    assert variant.env.actions == parent.env.actions
    assert variant.env.events == parent.env.events
    assert variant.env.terminations == parent.env.terminations
    assert variant.env.curriculum == parent.env.curriculum
    assert variant.env.scene.entities == parent.env.scene.entities
    assert variant.env.sim == parent.env.sim
    assert variant.env.viewer == parent.env.viewer
    assert variant.env.episode_length_s == parent.env.episode_length_s
    assert variant.env.seed == parent.env.seed
    assert [sensor.name for sensor in variant.env.scene.sensors] == [
      "contact_forces"
    ]

    parent_command = parent.env.commands["motion"]
    variant_command = variant.env.commands["motion"]
    assert type(variant_command) is type(parent_command)
    for field in fields(parent_command):
      if field.name in allowed_command_differences:
        continue
      assert getattr(variant_command, field.name) == getattr(
        parent_command, field.name
      ), (variant_name, field.name)

    assert tuple(variant_command.feet_standing_body_names) == (
      "left_ankle_roll_link",
      "right_ankle_roll_link",
    )
    assert len(variant_command.body_names) == 22
    assert set(variant.env.metrics) == {
      *set(parent.env.metrics),
      "substep_tracking_cache",
    }


def test_bfm_heft_reward_terms_match_complete_sp_heft_reward() -> None:
  sp_env = build_env_cfg(_compose("tracking_bfm_sp").task)
  expected_names = tuple(sp_env.rewards)
  assert len(expected_names) == 18

  for task_name in PARENT_BY_HEFT_REWARD:
    env = build_env_cfg(_compose(task_name).task)
    assert tuple(env.rewards) == expected_names
    for term_name, sp_term in sp_env.rewards.items():
      term = env.rewards[term_name]
      assert term.func is sp_term.func
      assert term.weight == sp_term.weight
      params = dict(term.params)
      params.pop("keypoint_specs", None)
      params.pop("toe_specs", None)
      assert params == sp_term.params


def test_heft_reward_manager_can_forward_every_configured_param() -> None:
  env = build_env_cfg(_compose("tracking_bfm_heft_reward").task)
  for term_name, term in env.rewards.items():
    callback = term.func.__call__ if isinstance(term.func, type) else term.func
    parameters = signature(callback).parameters
    accepts_extra = any(
      parameter.kind is Parameter.VAR_KEYWORD
      for parameter in parameters.values()
    )
    unsupported = set(term.params) - set(parameters)
    assert accepts_extra or not unsupported, (term_name, unsupported)


def test_bfm_heft_reward_uses_strict_semantic_hand_and_toe_geometry() -> None:
  cfg = _compose("tracking_bfm_heft_reward").task
  specs = {item.name: item for item in cfg.obs.semantic_keypoints.heft}
  assert specs["left_hand"].body_name == "left_wrist_yaw_link"
  assert tuple(specs["left_hand"].local_pos) == (0.116, 0.0, 0.0)
  assert specs["left_hand"].correction_body_name == "left_wrist_pitch_link"
  assert tuple(specs["left_hand"].correction_local_pos) == (0.005, 0.0, 0.0)
  assert specs["right_hand"].correction_body_name == "right_wrist_pitch_link"

  toes = {item.name: item for item in cfg.obs.semantic_keypoints.feet_toes}
  assert toes["left_toe"].body_name == "left_ankle_roll_link"
  assert toes["right_toe"].body_name == "right_ankle_roll_link"
  assert tuple(toes["left_toe"].local_pos) == (0.1, 0.0, 0.0)
  assert tuple(toes["right_toe"].local_pos) == (0.1, 0.0, 0.0)

  reward_params = {
    item.name: item.params for item in cfg.reward.rewards if "params" in item
  }
  assert "keypoint_specs" in reward_params["keypoint_pos_tracking"]
  assert "toe_specs" in reward_params["feet_air_time_ref_dense"]
