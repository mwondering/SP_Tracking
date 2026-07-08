import mujoco
from hydra import compose, initialize_config_module

from sp_tracking.config.build_env import build_env_cfg


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
