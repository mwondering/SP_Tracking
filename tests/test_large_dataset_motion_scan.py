from pathlib import Path
from types import SimpleNamespace

import pytest

from sp_tracking.tasks.tracking.mdp import multi_command_largedataset as large_dataset_module
from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  LargeDatasetMultiMotionCommand,
)



def _scan_command(**overrides):
  cfg_values = {
    "motion_scan_backend": "auto",
    "motion_scan_workers": 0,
    "motion_scan_fd_executable": "fd",
    "motion_scan_log_interval_s": 0.0,
  }
  cfg_values.update(overrides)
  command = object.__new__(LargeDatasetMultiMotionCommand)
  command.cfg = SimpleNamespace(**cfg_values)
  return command



def _touch(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"")



def test_motion_path_scan_prefers_fd_when_available(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  root = tmp_path / "motions"
  expected = [
    str(root / "b" / "motion.npz"),
    str(root / "a" / "MOTION.NPZ"),
  ]
  fd_path = str(tmp_path / "fd")
  calls: list[list[str]] = []

  monkeypatch.setattr(large_dataset_module.shutil, "which", lambda _: fd_path)

  class FakePopen:
    def __init__(
      self,
      cmd: list[str],
      *,
      stdout,
      stderr,
      text: bool,
    ) -> None:
      calls.append(cmd)
      assert stdout == large_dataset_module.subprocess.PIPE
      assert stderr == large_dataset_module.subprocess.PIPE
      assert text is True
      self.stdout = iter([path + "\n" for path in reversed(expected)])
      self.stderr = SimpleNamespace(read=lambda: "")

    def wait(self) -> int:
      return 0

  monkeypatch.setattr(large_dataset_module.subprocess, "Popen", FakePopen)

  command = _scan_command(motion_scan_workers=6)

  assert command._scan_motion_path(str(root)) == sorted(expected)
  stdout = capsys.readouterr().out
  assert "scan motion path start" in stdout
  assert "backend=fd" in stdout
  assert f"path={root}" in stdout
  assert "motions=2" in stdout
  assert calls
  assert calls[0][0] == fd_path
  assert "--hidden" in calls[0]
  assert "--no-ignore" in calls[0]
  assert calls[0][calls[0].index("--threads") + 1] == "6"
  assert calls[0][-2] == r"(?i)\.npz$"
  assert calls[0][-1] == str(root)



def test_motion_path_scan_falls_back_to_parallel_python_when_fd_missing(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  root = tmp_path / "motions"
  _touch(root / "a" / "motion.npz")
  _touch(root / "a" / "ignore.txt")
  _touch(root / "b" / "MOTION.NPZ")
  _touch(root / "root.npz")

  monkeypatch.setattr(large_dataset_module.shutil, "which", lambda _: None)
  original_collect = LargeDatasetMultiMotionCommand._collect_motion_files_os_walk
  scanned_roots: list[str] = []

  def collect_with_record(path: str) -> tuple[list[str], int, int]:
    scanned_roots.append(Path(path).name)
    return original_collect(path)

  monkeypatch.setattr(
    LargeDatasetMultiMotionCommand,
    "_collect_motion_files_os_walk",
    staticmethod(collect_with_record),
  )
  command = _scan_command(motion_scan_workers=2)

  assert command._scan_motion_path(str(root)) == [
    str(root / "a" / "motion.npz"),
    str(root / "b" / "MOTION.NPZ"),
    str(root / "root.npz"),
  ]
  assert sorted(scanned_roots) == ["a", "b"]



def test_motion_path_scan_requires_fd_when_backend_is_fd(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  monkeypatch.setattr(large_dataset_module.shutil, "which", lambda _: None)
  command = _scan_command(motion_scan_backend="fd")

  with pytest.raises(FileNotFoundError, match="executable not found"):
    command._scan_motion_path(str(tmp_path))
