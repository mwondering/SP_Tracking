from hydra import compose, initialize_config_dir
from hydra import initialize_config_module
from pathlib import Path

from sp_tracking.scripts.train import prepare_train_cfg


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(*overrides: str):
  with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
    return compose(config_name="train", overrides=list(overrides))


def test_prepare_train_cfg_applies_motion_path_override() -> None:
  cfg = _compose("motion_path=/dataset/motions", "task.num_envs=16")

  prepared = prepare_train_cfg(cfg)

  assert prepared.env.scene.num_envs == 16
  assert prepared.env.commands["motion"].motion_path == "/dataset/motions"
  assert prepared.agent.seed == 42


def test_prepare_train_cfg_applies_agent_overrides() -> None:
  cfg = _compose("agent.max_iterations=7", "agent.run_name=debug")

  prepared = prepare_train_cfg(cfg)

  assert prepared.agent.max_iterations == 7
  assert prepared.agent.run_name == "debug"


def test_packaged_hydra_config_composes() -> None:
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    cfg = compose(config_name="train", overrides=["task.num_envs=8"])

  prepared = prepare_train_cfg(cfg)

  assert prepared.env.scene.num_envs == 8


def test_root_and_packaged_yaml_configs_match() -> None:
  package_conf = Path(__file__).resolve().parents[1] / "src" / "sp_tracking" / "conf"
  root_files = sorted(CONF_DIR.rglob("*.yaml"))

  assert root_files
  for root_file in root_files:
    packaged_file = package_conf / root_file.relative_to(CONF_DIR)
    assert packaged_file.read_text() == root_file.read_text()
