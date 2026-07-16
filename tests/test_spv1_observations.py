from __future__ import annotations

import math
from types import SimpleNamespace

from hydra import compose, initialize_config_module
import mujoco
import torch

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.scripts.train import prepare_train_cfg
from sp_tracking.tasks.tracking.mdp import sp as sp_mdp
from sp_tracking.tasks.tracking.mdp import spv1
from sp_tracking.tasks.tracking.mdp import spv2


class _CommandManager:
  def __init__(self, command) -> None:
    self.command = command

  def get_term(self, name: str):
    assert name == "motion"
    return self.command


class _ReferenceCommand:
  def __init__(self, fields: dict[str, torch.Tensor]) -> None:
    self.fields = fields
    self.cfg = SimpleNamespace(body_names=("pelvis",))
    self.motion_anchor_body_index = 0

  def gather_reference(
    self, field_name: str, relative_steps: tuple[int, ...]
  ) -> torch.Tensor:
    indices = torch.as_tensor(relative_steps, dtype=torch.long)
    return self.fields[field_name].index_select(1, indices)


def _yaw_quat(angle: torch.Tensor) -> torch.Tensor:
  zeros = torch.zeros_like(angle)
  return torch.stack(
    (torch.cos(angle / 2.0), zeros, zeros, torch.sin(angle / 2.0)), dim=-1
  )


def _reference_env(
  global_yaw: float = 0.0,
  joint_count: int = 29,
  robot_relative_yaw: float = 0.0,
):
  steps = torch.arange(7, dtype=torch.float32)
  local_pos = torch.stack((steps, steps * 0.25, steps * 0.1), dim=-1)
  global_quat = _yaw_quat(torch.tensor(global_yaw))
  local_quat = _yaw_quat(steps * 0.1)
  quat = sp_mdp.quat_mul(global_quat.expand_as(local_quat), local_quat)
  pos = sp_mdp.quat_apply_inverse(
    sp_mdp._quat_conjugate(global_quat).expand_as(quat), local_pos
  )
  local_ang_vel = torch.stack(
    (steps * 0.01, steps * 0.02, torch.full_like(steps, 0.1)), dim=-1
  )
  ang_vel_w = sp_mdp.quat_apply_inverse(
    sp_mdp._quat_conjugate(global_quat).expand_as(quat), local_ang_vel
  )
  local_lin_vel = torch.stack(
    (1.0 + steps * 0.1, 0.25 + steps * 0.02, steps * 0.01), dim=-1
  )
  lin_vel_w = sp_mdp.quat_apply_inverse(
    sp_mdp._quat_conjugate(global_quat).expand_as(quat), local_lin_vel
  )
  joint_scale = torch.linspace(-1.0, 1.0, joint_count)
  joint_pos = steps[:, None] * joint_scale[None]
  joint_vel = joint_pos * 0.5
  fields = {
    "body_pos_w": pos[None, :, None, :],
    "body_quat_w": quat[None, :, None, :],
    "body_ang_vel_w": ang_vel_w[None, :, None, :],
    "body_lin_vel_w": lin_vel_w[None, :, None, :],
    "joint_pos": joint_pos[None],
    "joint_vel": joint_vel[None],
  }
  command = _ReferenceCommand(fields)
  robot_quat = sp_mdp.quat_mul(
    global_quat,
    _yaw_quat(torch.tensor(robot_relative_yaw)),
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    command_manager=_CommandManager(command),
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(root_link_quat_w=robot_quat.unsqueeze(0))
      )
    },
  )
  return env, command


def _compose(task_name: str):
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=[f"task={task_name}"])


def test_reference_commands_are_heading_invariant_and_have_spv1_dimensions() -> None:
  base, _ = _reference_env(global_yaw=0.0)
  rotated, _ = _reference_env(global_yaw=1.1)

  functions_and_dims = (
    (spv1.root_pos_command, 18),
    (spv1.root_ori_command, 36),
    (spv1.ref_joint_pos, 203),
    (spv1.ref_joint_vel, 203),
    (spv1.ref_projected_gravity, 21),
    (spv1.ref_base_ang_vel, 21),
  )
  for function, expected_dim in functions_and_dims:
    base_value = function(base, command_name="motion")
    rotated_value = function(rotated, command_name="motion")
    assert base_value.shape == (1, expected_dim)
    torch.testing.assert_close(base_value, rotated_value, atol=1.0e-5, rtol=1.0e-5)


def test_explicit_joint_error_reuses_newest_noisy_proprioception(monkeypatch) -> None:
  env, command = _reference_env(joint_count=2)
  data = SimpleNamespace(
    joint_pos=torch.tensor([[0.4, -0.3]]),
    joint_pos_biased=torch.tensor([[0.5, -0.2]]),
    default_joint_pos=torch.tensor([[0.1, 0.1]]),
    joint_vel=torch.tensor([[0.2, -0.4]]),
    default_joint_vel=torch.zeros((1, 2)),
  )
  env.scene = {"robot": SimpleNamespace(data=data)}
  calls = 0

  def _noise(value: torch.Tensor, std: float) -> torch.Tensor:
    nonlocal calls
    calls += 1
    return value + std

  monkeypatch.setattr(spv1.sp_mdp, "_uniform_noise", _noise)
  measured = spv1.joint_pos(
    env, command_name="motion", biased=True, noise_std=0.01
  )
  error = spv1.joint_pos_error(
    env, command_name="motion", biased=True, noise_std=0.01
  )

  assert calls == 1
  torch.testing.assert_close(error, command.fields["joint_pos"][:, 0] - measured)


def test_spv2_reference_terms_use_current_plus_four_future_steps() -> None:
  base, _ = _reference_env(global_yaw=0.0, robot_relative_yaw=0.35)
  rotated, _ = _reference_env(global_yaw=-0.8, robot_relative_yaw=0.35)

  functions_and_dims = (
    (spv2.root_pos_command, 12),
    (spv2.root_ori_command, 30),
    (spv2.ref_root_height, 5),
    (spv2.ref_root_lin_vel, 15),
    (spv2.ref_joint_pos, 145),
    (spv2.ref_joint_vel, 145),
    (spv2.ref_projected_gravity, 15),
    (spv2.ref_base_ang_vel, 15),
  )
  for function, expected_dim in functions_and_dims:
    base_value = function(base, command_name="motion")
    rotated_value = function(rotated, command_name="motion")
    assert base_value.shape == (1, expected_dim)
    torch.testing.assert_close(
      base_value, rotated_value, atol=1.0e-5, rtol=1.0e-5
    )


def test_spv2_root_orientation_command_contains_robot_heading_error() -> None:
  aligned, _ = _reference_env(robot_relative_yaw=0.0)
  yaw_error, _ = _reference_env(robot_relative_yaw=0.5)

  aligned_obs = spv2.root_ori_command(aligned, command_name="motion")
  yaw_error_obs = spv2.root_ori_command(yaw_error, command_name="motion")

  assert aligned_obs.shape == yaw_error_obs.shape == (1, 30)
  assert not torch.allclose(aligned_obs, yaw_error_obs)


def test_substep_cache_averages_joint_torque_sensor_measurements() -> None:
  joint_names = ("j0", "j1")
  asset_data = SimpleNamespace(
    joint_pos=torch.zeros((1, 2)),
    joint_vel=torch.zeros((1, 2)),
  )
  contact_data = SimpleNamespace(found=torch.zeros((1, 2)))
  scene = {
    "robot": SimpleNamespace(joint_names=joint_names, data=asset_data),
    "contact_forces": SimpleNamespace(
      primary_names=("left", "right"), data=contact_data
    ),
    "robot/tau_j0": SimpleNamespace(data=torch.zeros((1, 1))),
    "robot/tau_j1": SimpleNamespace(data=torch.zeros((1, 1))),
  }
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    physics_dt=0.005,
    common_step_counter=1,
    cfg=SimpleNamespace(decimation=4),
    scene=scene,
  )
  cache = sp_mdp.substep_tracking_cache(
    SimpleNamespace(
      params={
        "sensor_name": "contact_forces",
        "joint_torque_sensor_prefix": "tau_",
      }
    ),
    env,
  )
  samples = (
    (1.0, 10.0),
    (2.0, 20.0),
    (3.0, 30.0),
    (4.0, 40.0),
  )
  for left, right in samples:
    scene["robot/tau_j0"].data[:] = left
    scene["robot/tau_j1"].data[:] = right
    cache(env)

  torch.testing.assert_close(
    cache.joint_torque_average(), torch.tensor([[2.5, 25.0]])
  )
  torch.testing.assert_close(spv1.joint_torque(env), torch.tensor([[2.5, 25.0]]))


def test_spv1_task_has_exact_actor_critic_reward_and_sensor_contract() -> None:
  cfg = _compose("tracking_bfm_spv1_actor_heft_critic_heft_reward")
  prepared = prepare_train_cfg(cfg)
  env = build_env_cfg(cfg.task)
  actor = env.observations["actor"]

  assert tuple(actor.terms) == (
    "joint_pos",
    "joint_vel",
    "projected_gravity",
    "base_ang_vel",
    "last_action",
    "joint_torque",
    "root_pos_command",
    "root_ori_command",
    "ref_joint_pos",
    "ref_joint_vel",
    "ref_projected_gravity",
    "ref_base_ang_vel",
    "joint_pos_error",
    "joint_vel_error",
    "projected_gravity_error",
    "base_ang_vel_error",
  )
  assert all(
    actor.terms[name].history_length == 10
    for name in (
      "joint_pos",
      "joint_vel",
      "projected_gravity",
      "base_ang_vel",
      "last_action",
      "joint_torque",
    )
  )
  assert 10 * (4 * 29 + 2 * 3) + 18 + 36 + 2 * (7 * 29) + 2 * (7 * 3) + 64 == 1786
  assert prepared.agent.obs_groups == {
    "actor": ("actor",),
    "critic": ("policy", "priv"),
  }
  assert prepared.agent.critic.class_name.endswith(":HeftTeacherCritic")
  assert cfg.task.variant.reward_profile == "heft_tracking_bfm"
  assert "root_pos_tracking" in env.rewards
  assert env.metrics["substep_tracking_cache"].params[
    "joint_torque_sensor_prefix"
  ] == "spv1_joint_torque_"
  assert tuple(env.commands["motion"].reference_cache_steps["joint_vel"]) == tuple(
    range(7)
  )

  model = env.scene.entities["robot"].spec_fn().compile()
  torque_sensor_ids = [
    sensor_id
    for sensor_id in range(model.nsensor)
    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id).startswith(
      "spv1_joint_torque_"
    )
  ]
  assert len(torque_sensor_ids) == 29
  assert all(
    model.sensor_type[sensor_id] == mujoco.mjtSensor.mjSENS_JOINTACTFRC
    for sensor_id in torque_sensor_ids
  )


def test_spv2_has_compact_actor_and_heft_critic_contract() -> None:
  spv1_cfg = _compose("tracking_bfm_spv1_actor_heft_critic_heft_reward")
  cfg = _compose("tracking_bfm_spv2_actor_heft_critic_heft_reward")
  prepared = prepare_train_cfg(cfg)
  env = build_env_cfg(cfg.task)
  spv1_env = build_env_cfg(spv1_cfg.task)

  spv1_terms = tuple(spv1_env.observations["actor"].terms)
  spv2_terms = tuple(env.observations["actor"].terms)
  assert spv2_terms == (
    *spv1_terms[:8],
    "ref_root_height",
    "ref_root_lin_vel",
    *spv1_terms[8:],
  )
  actor = env.observations["actor"]
  assert all(
    actor.terms[name].history_length == 5
    for name in (
      "joint_pos",
      "joint_vel",
      "projected_gravity",
      "base_ang_vel",
      "last_action",
      "joint_torque",
    )
  )
  assert actor.terms["root_pos_command"].func is spv2.root_pos_command
  assert actor.terms["root_ori_command"].func is spv2.root_ori_command
  assert actor.terms["root_ori_command"].params["noise_std"] == 0.1
  assert actor.terms["ref_joint_pos"].func is spv2.ref_joint_pos
  assert actor.terms["ref_joint_vel"].func is spv2.ref_joint_vel
  assert actor.terms["ref_projected_gravity"].func is spv2.ref_projected_gravity
  assert actor.terms["ref_base_ang_vel"].func is spv2.ref_base_ang_vel
  assert (
    5 * (4 * 29 + 2 * 3)
    + 12
    + 30
    + 2 * (5 * 29)
    + 3 * (5 * 3)
    + 5
    + 64
    == 1056
  )
  assert prepared.agent.obs_groups == {
    "actor": ("actor",),
    "critic": ("policy", "priv"),
  }
  assert prepared.agent.critic.class_name.endswith(":HeftTeacherCritic")
  assert cfg.task.variant.reward_profile == "heft_tracking_bfm"
  assert tuple(env.commands["motion"].reference_cache_steps["joint_vel"]) == tuple(
    range(5)
  )
  expected_body_velocity_steps = (
    -8,
    -4,
    -2,
    -1,
    0,
    1,
    2,
    3,
    4,
    8,
    12,
    16,
    20,
  )
  assert tuple(
    env.commands["motion"].reference_cache_steps["body_lin_vel_w"]
  ) == expected_body_velocity_steps
  assert tuple(
    env.commands["motion"].reference_cache_steps["body_ang_vel_w"]
  ) == expected_body_velocity_steps
  assert (
    env.scene.entities["robot"].spec_fn
    is spv1_env.scene.entities["robot"].spec_fn
  )


def test_spv3_has_estimator_history_targets_and_heft_critic_contract() -> None:
  cfg = _compose("tracking_bfm_spv3_actor_heft_critic_heft_reward")
  prepared = prepare_train_cfg(cfg)
  env = build_env_cfg(cfg.task)

  assert tuple(env.observations) == (
    "policy",
    "priv",
    "actor_core",
    "estimator_history",
    "estimator_target",
  )
  assert tuple(env.observations["actor_core"].terms) == (
    "root_pos_command",
    "root_ori_command",
    "ref_root_height",
    "ref_root_lin_vel",
    "ref_joint_pos",
    "ref_joint_vel",
    "ref_projected_gravity",
    "ref_base_ang_vel",
    "joint_pos_error",
    "joint_vel_error",
    "projected_gravity_error",
    "base_ang_vel_error",
  )
  history = env.observations["estimator_history"]
  assert tuple(history.terms) == (
    "joint_pos",
    "joint_vel",
    "projected_gravity",
    "base_ang_vel",
    "last_action",
    "joint_torque",
  )
  assert all(term.history_length == 50 for term in history.terms.values())
  assert tuple(env.observations["estimator_target"].terms) == (
    "root_height",
    "root_lin_vel_b",
  )
  assert prepared.agent.obs_groups == {
    "actor": ("actor_core", "estimator_history"),
    "critic": ("policy", "priv"),
  }
  assert prepared.agent.actor.class_name.endswith(":SPV3EstimatorActor")
  assert prepared.agent.actor.estimator_hidden_dims == (512, 256, 128)
  assert prepared.agent.algorithm.class_name.endswith(":SPV3EstimatorPPO")
  assert prepared.agent.critic.class_name.endswith(":HeftTeacherCritic")


def test_spv4_adds_three_current_root_frame_key_body_groups() -> None:
  cfg = _compose("tracking_bfm_spv4_actor_heft_critic_heft_reward")
  prepared = prepare_train_cfg(cfg)
  env = build_env_cfg(cfg.task)

  assert tuple(env.observations) == (
    "policy",
    "priv",
    "actor_core",
    "estimator_history",
    "estimator_target",
    "robot_key_body",
    "ref_key_body",
    "key_body_error",
  )
  for name in ("robot_key_body", "ref_key_body", "key_body_error"):
    group = env.observations[name]
    assert tuple(group.terms) == ("current",)
    assert group.enable_corruption is False
    assert group.terms["current"].history_length == 0
    assert len(group.terms["current"].params["keypoint_specs"]) == 13
  assert prepared.agent.obs_groups == {
    "actor": (
      "actor_core",
      "estimator_history",
      "robot_key_body",
      "ref_key_body",
      "key_body_error",
    ),
    "critic": ("policy", "priv"),
  }
  assert prepared.agent.actor.class_name.endswith(":SPV4KeyBodyActor")
  assert prepared.agent.algorithm.class_name.endswith(":SPV3EstimatorPPO")
  assert prepared.agent.critic.class_name.endswith(":HeftTeacherCritic")
  assert cfg.task.command.command.fk_from_joint_pos is False
