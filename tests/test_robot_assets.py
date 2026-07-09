import mujoco
from hydra import compose, initialize_config_module

from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.assets.robots.safety import SAFETY_EFFORT_LIMITS


def _compose(*overrides: str):
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=list(overrides))


def _compiled_body_names(env_cfg) -> set[str]:
  robot = env_cfg.scene.entities["robot"]
  model = robot.spec_fn().compile()
  return {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    for body_id in range(model.nbody)
  }


def _compiled_actuator_limits(env_cfg) -> dict[str, float]:
  limits = {}
  for actuator in env_cfg.scene.entities["robot"].articulation.actuators:
    for expr in actuator.target_names_expr:
      limits[str(expr)] = float(actuator.effort_limit)
  return limits


def test_sp_body_names_exist_in_selected_robot_asset() -> None:
  cfg = _compose("task=tracking_bfm_sp")

  env_cfg = build_env_cfg(cfg.task)

  body_names = _compiled_body_names(env_cfg)
  assert set(env_cfg.commands["motion"].body_names) <= body_names


def test_tracking_bfm_body_names_exist_in_selected_robot_asset() -> None:
  cfg = _compose("task=tracking_bfm")

  env_cfg = build_env_cfg(cfg.task)

  robot = env_cfg.scene.entities["robot"]
  assert robot.spec_fn.__module__.startswith("sp_tracking.assets.robots.g1_tracking_bfm")
  body_names = _compiled_body_names(env_cfg)
  assert set(env_cfg.commands["motion"].body_names) <= body_names


def test_all_tasks_use_lowest_sim2real_torque_limits() -> None:
  for task in ("tracking_bfm", "tracking_bfm_largedataset", "tracking_bfm_sp"):
    cfg = _compose(f"task={task}")
    env_cfg = build_env_cfg(cfg.task)
    limits = _compiled_actuator_limits(env_cfg)

    for joint_name, expected in SAFETY_EFFORT_LIMITS.items():
      matched_limits = [
        limit
        for expr, limit in limits.items()
        if expr == joint_name
        or (expr.startswith(".*_") and joint_name.endswith(expr.removeprefix(".*")))
      ]
      assert matched_limits, (task, joint_name)
      assert min(matched_limits) == expected, (task, joint_name, matched_limits)
