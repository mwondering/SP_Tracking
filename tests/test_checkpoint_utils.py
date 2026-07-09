from pathlib import Path

import pytest

from sp_tracking.tasks.tracking.rl.checkpoints import (
  checkpoint_iteration,
  checkpoint_sort_key,
  resolve_local_checkpoint_path,
)


def test_checkpoint_iteration_accepts_model_names_and_final() -> None:
  assert checkpoint_iteration("model_42.pt") == 42
  assert checkpoint_iteration(Path("model_final.pt")) is None
  assert checkpoint_iteration("checkpoint_42.pt") == 42


def test_checkpoint_sort_key_keeps_model_final_last() -> None:
  names = ["model_9.pt", "model_final.pt", "model_100.pt", "model_10.pt"]

  assert sorted(names, key=checkpoint_sort_key) == [
    "model_9.pt",
    "model_10.pt",
    "model_100.pt",
    "model_final.pt",
  ]


def test_resolve_local_checkpoint_path_selects_latest_model_checkpoint(tmp_path: Path) -> None:
  log_root = tmp_path / "logs" / "rsl_rl" / "g1_tracking"
  first = log_root / "2026-07-09_10-00-00_a"
  second = log_root / "2026-07-09_11-00-00_b"
  first.mkdir(parents=True)
  second.mkdir(parents=True)
  (first / "model_100.pt").write_text("old")
  (second / "model_9.pt").write_text("new")
  (second / "model_final.pt").write_text("final")

  resolved = resolve_local_checkpoint_path(
    log_root=log_root,
    load_run=".*",
    load_checkpoint="model_.*.pt",
  )

  assert resolved == second / "model_final.pt"


def test_resolve_local_checkpoint_path_rejects_missing_match(tmp_path: Path) -> None:
  log_root = tmp_path / "logs" / "rsl_rl" / "g1_tracking"
  (log_root / "run").mkdir(parents=True)

  with pytest.raises(ValueError, match="No checkpoint found"):
    resolve_local_checkpoint_path(
      log_root=log_root,
      load_run=".*",
      load_checkpoint="model_.*.pt",
    )
