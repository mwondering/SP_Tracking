from hydra import compose, initialize_config_module
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from rsl_rl.utils.log_writer import LogWriter

from sp_tracking.scripts import train as train_module
from sp_tracking.scripts.train import (
  _copy_launch_script_to_log_dir,
  _resolve_resume_path,
  _resolve_runtime_device,
  _serialize_checkpoint_cfg,
  _save_resolved_cfg,
  prepare_train_cfg,
)
from sp_tracking.tasks.tracking.rl.runner import _upload_launch_script_artifact


def _compose(*overrides: str):
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
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
  cfg = _compose("task.num_envs=8")

  prepared = prepare_train_cfg(cfg)

  assert prepared.env.scene.num_envs == 8


def test_nonfinite_physics_tracer_is_opt_in_for_sp_task() -> None:
  sp_cfg = _compose("task=tracking_bfm_sp")
  baseline_cfg = _compose("task=tracking_bfm")
  debug_cfg = _compose(
    "task=tracking_bfm_sp", "task.debug_nonfinite_state=true"
  )

  assert sp_cfg.task.debug_nonfinite_state is False
  assert baseline_cfg.task.get("debug_nonfinite_state", False) is False
  assert debug_cfg.task.debug_nonfinite_state is True


def test_reference_cache_defaults_and_sp_step_groups() -> None:
  sp_cfg = _compose("task=tracking_bfm_sp")
  baseline_cfg = _compose("task=tracking_bfm")

  assert baseline_cfg.task.command.command.reference_cache_enabled is True
  assert baseline_cfg.task.command.command.reference_cache_steps is None
  assert sp_cfg.task.command.command.reference_cache_enabled is True
  assert tuple(sp_cfg.task.command.command.reference_cache_steps.joint_vel) == (0,)
  assert tuple(
    sp_cfg.task.command.command.reference_cache_steps.body_lin_vel_w
  ) == (-8, -4, -2, -1, 0, 1, 2, 4, 8, 12, 16, 20)


def test_sp_task_applies_one_stage_sp_tracking_agent_preset() -> None:
  cfg = _compose("task=tracking_bfm_sp")

  prepared = prepare_train_cfg(cfg)

  assert prepared.agent.num_steps_per_env == 32
  assert prepared.agent.actor.hidden_dims == (1024, 1024, 512)
  assert prepared.agent.critic.hidden_dims == (1024, 512, 512)
  assert prepared.agent.actor.class_name.endswith(":HeftTeacherActor")
  assert prepared.agent.critic.class_name.endswith(":HeftTeacherCritic")
  assert prepared.agent.algorithm.class_name.endswith(":HeftTeacherPPO")
  assert prepared.agent.actor.activation == "mish"
  assert prepared.agent.critic.activation == "mish"
  assert prepared.agent.obs_groups == {
    "actor": ("policy", "priv"),
    "critic": ("policy", "priv", "priv_critic"),
  }
  assert prepared.agent.algorithm.num_learning_epochs == 3
  assert prepared.agent.algorithm.num_mini_batches == 8
  assert prepared.agent.algorithm.actor_learning_rate == 1.0e-4
  assert prepared.agent.algorithm.critic_learning_rate == 5.0e-4
  assert prepared.agent.algorithm.clamp_rewards_min == 0.0
  assert prepared.agent.algorithm.optimizer == "muon"
  assert prepared.agent.algorithm.use_clipped_value_loss is False
  assert prepared.agent.algorithm.symmetry_cfg is None
  assert prepared.agent.save_interval == 150
  assert prepared.agent.seed == 0
  assert prepared.agent.load_checkpoint == "checkpoint_.*.pt"
  assert prepared.agent.upload_model is False


def test_sp_agent_preset_allows_standard_cli_agent_overrides() -> None:
  cfg = _compose("task=tracking_bfm_sp", "agent.num_steps_per_env=7")

  prepared = prepare_train_cfg(cfg)

  assert prepared.agent.num_steps_per_env == 7


def test_baseline_task_keeps_original_training_stack() -> None:
  prepared = prepare_train_cfg(_compose("task=tracking_bfm"))

  assert prepared.agent.actor.class_name == "MLPModel"
  assert prepared.agent.critic.class_name == "MLPModel"
  assert prepared.agent.algorithm.class_name == "PPO"
  assert prepared.agent.seed == 42


def test_root_conf_directory_is_not_a_second_config_source() -> None:
  root_conf = Path(__file__).resolve().parents[1] / "conf"

  assert not root_conf.exists()


def test_copy_launch_script_to_log_dir(tmp_path: Path) -> None:
  launch_script = tmp_path / "train_tracking_bfm.sh"
  launch_script.write_text("#!/usr/bin/env bash\nuv run sp-train\n")
  log_dir = tmp_path / "logs" / "run"

  copied = _copy_launch_script_to_log_dir(log_dir, str(launch_script))

  assert copied == log_dir / "launch" / "train_tracking_bfm.sh"
  assert copied.read_text() == launch_script.read_text()


def test_save_resolved_cfg_writes_cfg_and_config(tmp_path: Path) -> None:
  cfg = OmegaConf.create({"task": {"num_envs": 4}, "agent": {"run_name": "x"}})

  _save_resolved_cfg(tmp_path, cfg)

  assert (tmp_path / "cfg.yaml").exists()
  assert (tmp_path / "config.yaml").exists()
  assert "num_envs: 4" in (tmp_path / "cfg.yaml").read_text()


def test_serialize_checkpoint_cfg_is_a_resolved_mapping() -> None:
  cfg = OmegaConf.create({"task": {"name": "tracking_bfm"}, "agent": {"n": 4}})

  serialized = _serialize_checkpoint_cfg(cfg)

  assert serialized == {"task": {"name": "tracking_bfm"}, "agent": {"n": 4}}


def test_resolve_resume_path_uses_explicit_checkpoint_path(tmp_path: Path) -> None:
  checkpoint = tmp_path / "checkpoint_final.pt"
  checkpoint.write_bytes(b"ckpt")
  cfg = OmegaConf.create({"checkpoint_path": str(checkpoint)})
  agent_cfg = SimpleNamespace(resume=False, experiment_name="exp")

  assert _resolve_resume_path(cfg, agent_cfg) == checkpoint


def test_resolve_runtime_device_uses_local_rank_under_torchrun(monkeypatch) -> None:
  monkeypatch.setenv("WORLD_SIZE", "4")
  monkeypatch.setenv("RANK", "2")
  monkeypatch.setenv("LOCAL_RANK", "2")
  monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)

  device, rank, world_size = _resolve_runtime_device([0, 1, 2, 3])

  assert device == "cuda:2"
  assert rank == 2
  assert world_size == 4
  assert "CUDA_VISIBLE_DEVICES" not in __import__("os").environ
  assert __import__("os").environ["MUJOCO_EGL_DEVICE_ID"] == "2"


def test_prepare_train_cfg_keeps_num_envs_per_rank() -> None:
  cfg = _compose("task.num_envs=32")

  prepared = prepare_train_cfg(cfg)

  assert prepared.env.scene.num_envs == 32


def test_run_train_configures_torch_backends_before_env(monkeypatch, tmp_path: Path) -> None:
  events: list[str] = []
  prepared = SimpleNamespace(
    env=SimpleNamespace(),
    agent=SimpleNamespace(
      clip_actions=None,
      experiment_name="exp",
      max_iterations=0,
      run_name="",
      resume=False,
    ),
  )

  monkeypatch.setattr(train_module, "prepare_train_cfg", lambda cfg: prepared)
  monkeypatch.setattr(train_module, "_resolve_runtime_device", lambda gpu_ids: ("cpu", 0, 1))
  monkeypatch.setattr(train_module, "_make_log_dir", lambda cfg, agent: tmp_path / "run")
  monkeypatch.setattr(train_module, "_copy_launch_script_to_log_dir", lambda log_dir, path: None)
  monkeypatch.setattr(train_module, "_asdict_dataclass", lambda obj: {})
  monkeypatch.setattr(
    train_module,
    "configure_torch_backends",
    lambda: events.append("torch"),
    raising=False,
  )

  class FakeEnv:
    def __init__(self, *args, **kwargs) -> None:
      events.append("env")

  class FakeWrapper:
    def __init__(self, env, clip_actions=None) -> None:
      events.append("wrapper")

    def close(self) -> None:
      events.append("close")

  class FakeRunner:
    def __init__(self, *args, **kwargs) -> None:
      events.append("runner")

    def learn(self, *args, **kwargs) -> None:
      events.append("learn")

  monkeypatch.setattr(train_module, "ManagerBasedRlEnv", FakeEnv)
  monkeypatch.setattr(train_module, "RslRlVecEnvWrapper", FakeWrapper)
  monkeypatch.setattr(train_module, "SpTrackingOnPolicyRunner", FakeRunner)

  train_module.run_train(OmegaConf.create({"gpu_ids": [0]}))

  assert events[:2] == ["torch", "env"]


class _FakeWriter(LogWriter):
  def __init__(self) -> None:
    self.saved_files: list[str] = []

  def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
    return None

  def save_file(self, path: str) -> None:
    self.saved_files.append(path)


class _FakeLogger:
  def __init__(self) -> None:
    self.writer = _FakeWriter()


def test_upload_launch_script_artifact_uses_log_writer(tmp_path: Path) -> None:
  launch_script = tmp_path / "launch" / "train_tracking_bfm.sh"
  launch_script.parent.mkdir()
  launch_script.write_text("#!/usr/bin/env bash\n")
  logger = _FakeLogger()

  uploaded = _upload_launch_script_artifact(logger, launch_script)

  assert uploaded is True
  assert logger.writer.saved_files == [str(launch_script)]
