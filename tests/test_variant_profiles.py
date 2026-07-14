from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg


ABLATION_TASKS = {
  "tracking_bfm_sp_ablation_bfm_actor": ("actor",),
  "tracking_bfm_sp_ablation_student_actor": ("policy",),
  "tracking_bfm_sp_ablation_teacher_actor": ("policy", "priv"),
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
      "priv_critic",
    )
    assert prepared.agent.obs_groups == {
      "actor": actor_groups,
      "critic": ("policy", "priv", "priv_critic"),
    }


def test_actor_observation_is_the_only_difference_between_ablation_agents() -> None:
  prepared = {
    name: prepare_train_cfg(_compose(name)) for name in ABLATION_TASKS
  }
  first = prepared["tracking_bfm_sp_ablation_bfm_actor"].agent

  for item in prepared.values():
    agent = item.agent
    assert agent.actor.class_name == "MLPModel"
    assert agent.critic.class_name == "MLPModel"
    assert agent.algorithm.class_name == "PPO"
    assert agent.actor.hidden_dims == first.actor.hidden_dims
    assert agent.critic.hidden_dims == first.critic.hidden_dims
    assert agent.actor.activation == first.actor.activation == "elu"
    assert agent.critic.activation == first.critic.activation == "elu"
    assert agent.algorithm == first.algorithm
    assert agent.num_steps_per_env == first.num_steps_per_env == 24
    assert agent.seed == first.seed == 42

    actual = asdict(agent)
    expected = asdict(first)
    actual.pop("obs_groups")
    expected.pop("obs_groups")
    assert actual == expected


def test_ablation_observation_terms_are_reused_without_adapter() -> None:
  bfm = _compose("tracking_bfm")
  sp = _compose("tracking_bfm_sp")
  expected_actor = OmegaConf.to_container(
    bfm.task.obs.observations.actor, resolve=True
  )

  for task_name in ABLATION_TASKS:
    cfg = _compose(task_name)
    assert OmegaConf.to_container(
      cfg.task.obs.observations.actor, resolve=True
    ) == expected_actor
    for group in ("policy", "priv", "priv_critic"):
      assert OmegaConf.to_container(
        cfg.task.obs.observations[group], resolve=True
      ) == OmegaConf.to_container(sp.task.obs.observations[group], resolve=True)
    assert "adapter" not in OmegaConf.to_container(cfg.task, resolve=True)


def test_ablation_runtime_matches_bfm_except_sp_xml_compatibility() -> None:
  expected_events = {
    "push_robot",
    "base_com",
    "base_mass",
    "encoder_bias",
    "foot_friction",
  }
  expected_terminations = {"time_out", "anchor_pos", "anchor_ori", "ee_body_pos"}

  for task_name in ABLATION_TASKS:
    cfg = _compose(task_name)
    env = build_env_cfg(cfg.task)
    command = env.commands["motion"]
    robot = env.scene.entities["robot"]

    assert robot.spec_fn.__module__.startswith(
      "sp_tracking.assets.robots.g1_sp_tracking"
    )
    assert type(env.actions["joint_pos"]).__name__ == "JointPositionActionCfg"
    assert type(command).__name__ == "MultiMotionCommandCfg"
    assert command.motion_type == "mujoco"
    assert command.fk_from_joint_pos is True
    assert command.recompute_joint_vel_from_joint_pos is True
    assert command.sampling_mode == "adaptive"
    assert command.rewind.enabled is False
    assert command.feet_standing == {}
    assert command.resample_on_motion_end is True
    assert set(env.events) == expected_events
    assert set(env.terminations) == expected_terminations
    assert env.curriculum == {}
    assert env.metrics == {}
    assert "motion_global_root_pos" in env.rewards
    assert "root_pos_tracking" not in env.rewards
    assert env.episode_length_s == 10.0
    assert env.viewer.body_name == "torso_link"
    assert {sensor.name for sensor in env.scene.sensors} == {
      "contact_forces",
      "self_collision",
    }
    assert tuple(command.body_names) == tuple(
      cfg.task.reference_views.combined.body_names
    )
