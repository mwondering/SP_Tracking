from __future__ import annotations

import re
from pathlib import Path

from hydra import compose, initialize_config_module
from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from omegaconf import OmegaConf

from sp_tracking.assets.robots.g1_sp_tracking import get_g1_sp_tracking_spec
from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.task_catalog import TASK_SPECS


ABLATION_TASKS = {
  "tracking_bfm_sp_ablation_bfm_actor": ("actor",),
  "tracking_bfm_sp_ablation_student_actor": ("policy",),
  "tracking_bfm_sp_ablation_teacher_actor": ("policy", "priv"),
}

BFM_CRITIC_BASELINES = {
  "tracking_bfm_student_actor_bfm_critic": ("policy",),
  "tracking_bfm_teacher_actor_bfm_critic": ("policy", "priv"),
}

WBTELEOP_TASKS = (
  "tracking_bfm_wbteleop_actor_bfm_critic",
  "tracking_bfm_wbteleop_actor_heft_critic",
)

EXPECTED_TASK_SEMANTICS = {
  "tracking_bfm": (
    ("actor", "critic"),
    ("actor",),
    ("critic",),
    "MLPModel",
    "MLPModel",
    "JointPositionActionCfg",
    "old_tracking",
  ),
  "tracking_bfm_sp": (
    ("policy", "priv", "priv_critic"),
    ("policy", "priv"),
    ("policy", "priv", "priv_critic"),
    "sp_tracking.tasks.tracking.rl.heft_models:HeftTeacherActor",
    "sp_tracking.tasks.tracking.rl.heft_models:HeftTeacherCritic",
    "SpTrackingJointPositionActionCfg",
    "sp_tracking",
  ),
  "tracking_bfm_sp_ablation_bfm_actor": (
    ("actor", "policy", "priv"),
    ("actor",),
    ("policy", "priv"),
    "MLPModel",
    "sp_tracking.tasks.tracking.rl.heft_models:HeftTeacherCritic",
    "ObservationHistoryJointPositionActionCfg",
    "old_tracking",
  ),
  "tracking_bfm_sp_ablation_student_actor": (
    ("actor", "policy", "priv"),
    ("policy",),
    ("policy", "priv"),
    "MLPModel",
    "sp_tracking.tasks.tracking.rl.heft_models:HeftTeacherCritic",
    "ObservationHistoryJointPositionActionCfg",
    "old_tracking",
  ),
  "tracking_bfm_sp_ablation_teacher_actor": (
    ("actor", "policy", "priv"),
    ("policy", "priv"),
    ("policy", "priv"),
    "MLPModel",
    "sp_tracking.tasks.tracking.rl.heft_models:HeftTeacherCritic",
    "ObservationHistoryJointPositionActionCfg",
    "old_tracking",
  ),
  "tracking_bfm_student_actor_bfm_critic": (
    ("actor", "critic", "policy", "priv"),
    ("policy",),
    ("critic",),
    "MLPModel",
    "MLPModel",
    "ObservationHistoryJointPositionActionCfg",
    "old_tracking",
  ),
  "tracking_bfm_teacher_actor_bfm_critic": (
    ("actor", "critic", "policy", "priv"),
    ("policy", "priv"),
    ("critic",),
    "MLPModel",
    "MLPModel",
    "ObservationHistoryJointPositionActionCfg",
    "old_tracking",
  ),
  "tracking_bfm_wbteleop_actor_bfm_critic": (
    ("actor", "critic"),
    ("actor",),
    ("critic",),
    "MLPModel",
    "MLPModel",
    "JointPositionActionCfg",
    "old_tracking",
  ),
  "tracking_bfm_wbteleop_actor_heft_critic": (
    ("actor", "policy", "priv"),
    ("actor",),
    ("policy", "priv"),
    "MLPModel",
    "sp_tracking.tasks.tracking.rl.heft_models:HeftTeacherCritic",
    "ObservationHistoryJointPositionActionCfg",
    "old_tracking",
  ),
}


def _compose(task_name: str):
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=[f"task={task_name}"])


def test_ablation_base_inherits_tracking_bfm_directly() -> None:
  config_path = (
    Path(__file__).parents[1]
    / "src/sp_tracking/conf/task/tracking_bfm_sp_ablation_bfm_actor.yaml"
  )
  source = OmegaConf.load(config_path)
  assert source.defaults[0] == "tracking_bfm"
  assert "tracking_bfm_sp" not in source.defaults


def test_supported_task_profile_matrix() -> None:
  bfm = prepare_train_cfg(_compose("tracking_bfm"))
  sp = prepare_train_cfg(_compose("tracking_bfm_sp"))

  assert tuple(bfm.env.observations) == ("actor", "critic")
  assert bfm.agent.obs_groups == {"actor": ("actor",), "critic": ("critic",)}
  assert tuple(sp.env.observations) == ("policy", "priv", "priv_critic")
  assert sp.agent.obs_groups == {
    "actor": ("policy", "priv"),
    "critic": ("policy", "priv", "priv_critic"),
  }

  for task_name, actor_groups in ABLATION_TASKS.items():
    prepared = prepare_train_cfg(_compose(task_name))
    assert tuple(prepared.env.observations) == (
      "actor",
      "policy",
      "priv",
    )
    assert prepared.agent.obs_groups == {
      "actor": actor_groups,
      "critic": ("policy", "priv"),
    }


def test_every_catalog_task_matches_declared_semantics() -> None:
  assert {spec.name for spec in TASK_SPECS} == set(EXPECTED_TASK_SEMANTICS)

  for task_name, expected in EXPECTED_TASK_SEMANTICS.items():
    (
      env_groups,
      actor_groups,
      critic_groups,
      actor_class,
      critic_class,
      action_class,
      reward_profile,
    ) = expected
    prepared = prepare_train_cfg(_compose(task_name))
    variant = prepared.raw.task.variant

    assert tuple(prepared.env.observations) == env_groups
    assert prepared.agent.obs_groups == {
      "actor": actor_groups,
      "critic": critic_groups,
    }
    assert prepared.agent.actor.class_name == actor_class
    assert prepared.agent.critic.class_name == critic_class
    assert type(prepared.env.actions["joint_pos"]).__name__ == action_class
    assert variant.reward_profile == reward_profile
    assert prepared.env.commands["motion"].anchor_body_name == "pelvis"


def test_actor_observation_is_the_only_difference_between_ablation_agents() -> None:
  prepared = {
    name: prepare_train_cfg(_compose(name)) for name in ABLATION_TASKS
  }
  first = prepared["tracking_bfm_sp_ablation_bfm_actor"].agent

  for item in prepared.values():
    agent = item.agent
    assert agent.actor.class_name == "MLPModel"
    assert agent.critic.class_name.endswith(":HeftTeacherCritic")
    assert agent.algorithm.class_name == "PPO"
    assert agent.critic.hidden_dims == first.critic.hidden_dims
    assert agent.actor.activation == first.actor.activation == "elu"
    assert agent.critic.activation == first.critic.activation == "mish"
    assert agent.critic.hidden_dims == (1024, 512, 512)
    assert agent.critic.vecnorm_decay == 0.9999
    assert agent.algorithm == first.algorithm
    assert agent.num_steps_per_env == first.num_steps_per_env == 24
    assert agent.seed == first.seed == 42

  actor_dims = {
    "tracking_bfm_sp_ablation_bfm_actor": 286,
    "tracking_bfm_sp_ablation_student_actor": 1728,
    "tracking_bfm_sp_ablation_teacher_actor": 6330,
  }

  def parameter_count(input_dim: int, hidden_dims: tuple[int, ...]) -> int:
    dims = (input_dim, *hidden_dims, 29)
    return sum(
      in_dim * out_dim + out_dim
      for in_dim, out_dim in zip(dims, dims[1:], strict=False)
    )

  counts = {
    name: parameter_count(actor_dims[name], item.agent.actor.hidden_dims)
    for name, item in prepared.items()
  }
  baseline_count = counts["tracking_bfm_sp_ablation_bfm_actor"]
  assert baseline_count == 8_624_669
  assert all(
    abs(count - baseline_count) / baseline_count < 0.001
    for count in counts.values()
  )


def test_student_and_teacher_actor_baselines_restore_bfm_critic() -> None:
  bfm = prepare_train_cfg(_compose("tracking_bfm"))

  for task_name, actor_groups in BFM_CRITIC_BASELINES.items():
    prepared = prepare_train_cfg(_compose(task_name))
    assert tuple(prepared.env.observations) == (
      "actor",
      "critic",
      "policy",
      "priv",
    )
    assert prepared.agent.obs_groups == {
      "actor": actor_groups,
      "critic": ("critic",),
    }
    assert prepared.agent.critic == bfm.agent.critic
    assert prepared.agent.algorithm == bfm.agent.algorithm
    assert prepared.agent.num_steps_per_env == bfm.agent.num_steps_per_env
    assert prepared.agent.seed == bfm.agent.seed


def test_bfm_critic_baselines_only_change_actor_observation_from_parent() -> None:
  parent_by_baseline = {
    "tracking_bfm_student_actor_bfm_critic": (
      "tracking_bfm_sp_ablation_student_actor"
    ),
    "tracking_bfm_teacher_actor_bfm_critic": (
      "tracking_bfm_sp_ablation_teacher_actor"
    ),
  }

  for baseline_name, parent_name in parent_by_baseline.items():
    baseline = _compose(baseline_name)
    parent = _compose(parent_name)
    for group in ("policy", "priv"):
      assert OmegaConf.to_container(
        baseline.task.obs.observations[group], resolve=True
      ) == OmegaConf.to_container(
        parent.task.obs.observations[group], resolve=True
      )
    assert baseline.task.obs.observations.critic.enabled is True
    assert parent.task.obs.observations.critic.enabled is False


def test_wbteleop_critic_pair_has_identical_actor_and_bfm_objective() -> None:
  bfm_critic = _compose(WBTELEOP_TASKS[0])
  heft_critic = _compose(WBTELEOP_TASKS[1])

  assert OmegaConf.to_container(
    bfm_critic.task.obs.observations.actor, resolve=True
  ) == OmegaConf.to_container(
    heft_critic.task.obs.observations.actor, resolve=True
  )
  for section in ("reward", "terminations", "events"):
    assert OmegaConf.to_container(
      bfm_critic.task[section], resolve=True
    ) == OmegaConf.to_container(
      heft_critic.task[section], resolve=True
    )

  bfm_prepared = prepare_train_cfg(bfm_critic)
  heft_prepared = prepare_train_cfg(heft_critic)
  assert bfm_prepared.agent.actor == heft_prepared.agent.actor
  assert bfm_prepared.agent.algorithm == heft_prepared.agent.algorithm
  assert bfm_prepared.agent.obs_groups == {
    "actor": ("actor",),
    "critic": ("critic",),
  }
  assert heft_prepared.agent.obs_groups == {
    "actor": ("actor",),
    "critic": ("policy", "priv"),
  }
  assert heft_prepared.agent.critic.class_name.endswith(":HeftTeacherCritic")
  assert heft_prepared.agent.critic.hidden_dims == (1024, 512, 512)
  assert heft_prepared.agent.critic.activation == "mish"
  assert heft_prepared.agent.critic.vecnorm_decay == 0.9999

  bfm_action = bfm_prepared.env.actions["joint_pos"]
  heft_action = heft_prepared.env.actions["joint_pos"]
  assert type(bfm_action).__name__ == "JointPositionActionCfg"
  assert type(heft_action).__name__ == "ObservationHistoryJointPositionActionCfg"
  assert heft_action.observation_history_steps == 8
  assert heft_action.scale == bfm_action.scale
  assert heft_action.use_default_offset == bfm_action.use_default_offset


def test_ablation_observation_terms_are_reused_without_adapter() -> None:
  bfm = _compose("tracking_bfm")
  sp = _compose("tracking_bfm_sp")
  expected_actor = OmegaConf.to_container(
    bfm.task.obs.observations.actor, resolve=True
  )

  for task_name in (*ABLATION_TASKS, *BFM_CRITIC_BASELINES):
    cfg = _compose(task_name)
    assert OmegaConf.to_container(
      cfg.task.obs.observations.actor, resolve=True
    ) == expected_actor
    for group in ("policy", "priv"):
      ablation_terms = cfg.task.obs.observations[group].terms
      sp_terms = sp.task.obs.observations[group].terms
      assert [(item.name, item.term) for item in ablation_terms] == [
        (item.name, item.term) for item in sp_terms
      ]
    assert tuple(cfg.task.obs.observations.policy.disabled_terms) == (
      "boot_indicator_state_obs",
    )
    assert tuple(cfg.task.obs.observations.priv.disabled_terms) == (
      "body_z_termination_obs",
      "gravity_dir_termination_obs",
    )
    assert cfg.task.obs.observations.priv_critic.enabled is False
    assert OmegaConf.to_container(
      cfg.task.obs.observations.priv_critic.terms, resolve=True
    ) == OmegaConf.to_container(
      sp.task.obs.observations.priv_critic.terms, resolve=True
    )
    assert "adapter" not in OmegaConf.to_container(cfg.task, resolve=True)
    assert cfg.task.reference_views.sp_tracking.anchor_body_name == "pelvis"
    semantic_names = tuple(item.name for item in cfg.task.obs.semantic_keypoints.heft)
    assert semantic_names == tuple(
      item.name for item in sp.task.obs.semantic_keypoints.heft
    )


def test_ablation_runtime_matches_bfm_with_semantic_keypoints_only() -> None:
  expected_events = {
    "push_robot",
    "base_com",
    "base_mass",
    "encoder_bias",
    "foot_friction",
  }
  expected_terminations = {"time_out", "anchor_pos", "anchor_ori", "ee_body_pos"}

  for task_name in (*ABLATION_TASKS, *BFM_CRITIC_BASELINES):
    cfg = _compose(task_name)
    env = build_env_cfg(cfg.task)
    command = env.commands["motion"]
    robot = env.scene.entities["robot"]
    baseline_robot = build_env_cfg(_compose("tracking_bfm").task).scene.entities[
      "robot"
    ]

    assert robot.spec_fn.__module__.startswith(
      "sp_tracking.assets.robots.g1_tracking_bfm"
    )
    assert robot.init_state == baseline_robot.init_state
    assert robot.articulation == baseline_robot.articulation
    assert robot.collisions == baseline_robot.collisions
    assert len(robot.joint_name_order) == 29
    assert not hasattr(robot, "joint_symmetry_mapping")
    assert not hasattr(robot, "spatial_symmetry_mapping")
    action = env.actions["joint_pos"]
    assert type(action).__name__ == "ObservationHistoryJointPositionActionCfg"
    assert action.observation_history_steps == 8
    assert action.scale == G1_ACTION_SCALE
    assert action.use_default_offset is True
    assert type(command).__name__ == "MultiMotionCommandCfg"
    assert command.anchor_body_name == "pelvis"
    assert command.motion_type == "isaaclab"
    assert command.fk_from_joint_pos is False
    assert command.recompute_joint_vel_from_joint_pos is False
    assert command.motion_origin_recenter is False
    assert command.sampling_mode == "adaptive"
    assert command.rewind.enabled is False
    assert tuple(command.feet_standing_body_names) == (
      "left_ankle_roll_link",
      "right_ankle_roll_link",
    )
    assert command.feet_standing["z_enter"] == 0.18
    assert command.resample_on_motion_end is True
    assert set(env.events) == expected_events
    assert set(env.terminations) == expected_terminations
    assert env.curriculum == {}
    assert set(env.metrics) == {"substep_tracking_cache"}
    assert env.metrics["substep_tracking_cache"].per_substep is True
    assert "motion_global_root_pos" in env.rewards
    assert "root_pos_tracking" not in env.rewards
    assert env.episode_length_s == 10.0
    assert env.viewer.body_name == "torso_link"
    assert env.sim.contact_sensor_maxmatch == 64
    assert {sensor.name for sensor in env.scene.sensors} == {
      "contact_forces",
      "self_collision",
    }
    assert tuple(command.body_names) == tuple(
      cfg.task.reference_views.combined.body_names
    )
    serialized = str(OmegaConf.to_container(cfg.task, resolve=True))
    assert "head_mimic" not in serialized
    assert "hand_mimic" not in serialized
    assert "toe_link" not in serialized
    assert "sp_xml_bfm_runtime_g1" not in serialized


def test_all_task_reward_and_observation_anchors_use_pelvis() -> None:
  task_names = (
    "tracking_bfm",
    "tracking_bfm_sp",
    *ABLATION_TASKS,
    *BFM_CRITIC_BASELINES,
    *WBTELEOP_TASKS,
  )
  for task_name in task_names:
    cfg = _compose(task_name)
    env = build_env_cfg(cfg.task)
    assert cfg.task.robot.anchor_body_name == "pelvis"
    assert env.commands["motion"].anchor_body_name == "pelvis"

    for view in cfg.task.reference_views.values():
      if "anchor_body_name" in view:
        assert view.anchor_body_name == "pelvis"

    for reward_cfg in env.rewards.values():
      anchor = reward_cfg.params.get("anchor_body_name")
      if anchor is not None:
        assert anchor == "pelvis"

    for group_cfg in env.observations.values():
      for term_cfg in group_cfg.terms.values():
        for param_name in ("anchor_body_name", "root_body_name"):
          anchor = term_cfg.params.get(param_name)
          if anchor is not None:
            assert anchor == "pelvis"


def test_sp_xml_foot_friction_pattern_matches_all_foot_collision_geoms() -> None:
  spec = get_g1_sp_tracking_spec()
  geom_names = tuple(geom.name for geom in spec.geoms)
  expected = {
    f"{side}_foot{index}_collision"
    for side in ("left", "right")
    for index in range(1, 8)
  }

  for task_name in (*ABLATION_TASKS, "tracking_bfm_sp"):
    cfg = _compose(task_name)
    pattern = str(cfg.task.robot.foot_geom_pattern)
    matched = {name for name in geom_names if re.fullmatch(pattern, name)}
    assert matched == expected
