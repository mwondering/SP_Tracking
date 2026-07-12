from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import DictConfig, OmegaConf
from rsl_rl.modules.normalization import EmpiricalNormalization

from sp_tracking.tasks.tracking.rl.runner import SpTrackingOnPolicyRunner


class _FakePolicy:
  obs_groups = ("actor",)

  def __init__(self) -> None:
    self.obs_normalizer = EmpiricalNormalization(2)
    with torch.no_grad():
      self.obs_normalizer._mean.fill_(2.0)
      self.obs_normalizer._var.fill_(3.0)
      self.obs_normalizer._std.fill_(3.0**0.5)
      self.obs_normalizer.count.fill_(5)


class _FakeAlg:
  def __init__(self):
    self.policy = _FakePolicy()
    self.loaded = None

  def save(self):
    return {
      "actor_state_dict": {"actor": torch.tensor([1.0])},
      "critic_state_dict": {"critic": torch.tensor([2.0])},
      "optimizer_state_dict": {"optimizer": torch.tensor([3.0])},
    }

  def get_policy(self):
    return self.policy

  def load(self, loaded_dict, load_cfg, strict):
    self.loaded = (loaded_dict, load_cfg, strict)
    return True


class _FakeWriter:
  def __init__(self):
    self.models = []
    self.files = []

  def save_model(self, path: str, it: int) -> None:
    self.models.append((Path(path).name, it))

  def save_file(self, path: str) -> None:
    self.files.append(Path(path).name)


class _FakeLogger:
  logger_type = "tensorboard"

  def __init__(self, log_dir: Path):
    self.writer = _FakeWriter()
    self.log_dir = str(log_dir)
    self.models = []

  def save_model(self, path: str, it: int) -> None:
    self.models.append((Path(path).name, it))


def _make_runner(tmp_path: Path) -> SpTrackingOnPolicyRunner:
  runner = object.__new__(SpTrackingOnPolicyRunner)
  runner.alg = _FakeAlg()
  runner.current_learning_iteration = 7
  runner.cfg = {"upload_model": True}
  runner.checkpoint_cfg = OmegaConf.create(
    {"task": {"name": "tracking_bfm_sp"}, "agent": {"run_name": "unit"}}
  )
  runner.logger = _FakeLogger(tmp_path)
  runner.env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      common_step_counter=123,
      command_manager=SimpleNamespace(get_term=lambda name: None),
    ),
    num_actions=2,
  )
  runner.log_dir = str(tmp_path)
  return runner


def test_runner_save_writes_sp_tracking_payload(tmp_path: Path, monkeypatch) -> None:
  runner = _make_runner(tmp_path)
  monkeypatch.setattr(runner, "_export_deploy_artifacts", lambda path: None)

  runner.save(str(tmp_path / "checkpoint_7.pt"))

  checkpoint = torch.load(tmp_path / "checkpoint_7.pt", weights_only=False)
  assert checkpoint["policy"] == {"actor": torch.tensor([1.0])}
  assert checkpoint["rsl_rl"]["critic_state_dict"] == {"critic": torch.tensor([2.0])}
  assert checkpoint["iter"] == 7
  assert checkpoint["env"]["common_step_counter"] == 123
  assert checkpoint["infos"]["env_state"]["common_step_counter"] == 123
  assert isinstance(checkpoint["cfg"], DictConfig)
  assert OmegaConf.to_container(checkpoint["cfg"], resolve=True) == {
    "task": {"name": "tracking_bfm_sp"},
    "agent": {"run_name": "unit"},
  }
  extra_state = checkpoint["vecnorm"]["_extra_state"]
  assert set(extra_state) == {
    "actor_sum",
    "actor_ssq",
    "actor_count",
    "policy_sum",
    "policy_ssq",
    "policy_count",
  }
  torch.testing.assert_close(extra_state["policy_sum"], torch.full((2,), 10.0))
  torch.testing.assert_close(extra_state["policy_ssq"], torch.full((2,), 35.0))
  torch.testing.assert_close(extra_state["policy_count"], torch.tensor([5.0]))


def test_runner_save_exports_fixed_policy_onnx_name(tmp_path: Path, monkeypatch) -> None:
  runner = _make_runner(tmp_path)
  exported = []
  monkeypatch.setattr(
    runner,
    "export_policy_to_onnx",
    lambda path, filename="policy.onnx", verbose=False: exported.append((Path(path), filename)),
  )

  runner.save(str(tmp_path / "checkpoint_7.pt"))

  assert exported == [(tmp_path, "policy.onnx")]


def test_runner_load_reads_sp_tracking_payload(tmp_path: Path) -> None:
  runner = _make_runner(tmp_path)
  checkpoint = {
    "policy": {"actor": torch.tensor([4.0])},
    "env": {"common_step_counter": 456},
    "rsl_rl": {
      "actor_state_dict": {"actor": torch.tensor([4.0])},
      "critic_state_dict": {"critic": torch.tensor([5.0])},
      "optimizer_state_dict": {"optimizer": torch.tensor([6.0])},
    },
    "iter": 99,
    "infos": {"extra": "value"},
  }
  torch.save(checkpoint, tmp_path / "checkpoint_99.pt")

  infos = runner.load(str(tmp_path / "checkpoint_99.pt"), map_location="cpu")

  loaded_dict, load_cfg, strict = runner.alg.loaded
  assert loaded_dict["actor_state_dict"] == {"actor": torch.tensor([4.0])}
  assert load_cfg is None
  assert strict is True
  assert runner.current_learning_iteration == 99
  assert runner.env.unwrapped.common_step_counter == 456
  assert infos == {"extra": "value"}
