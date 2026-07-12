from pathlib import Path

import pytest

from sp_tracking.tasks.tracking.rl.checkpoints import (
  checkpoint_iteration,
  checkpoint_sort_key,
  resolve_local_checkpoint_path,
)


def test_checkpoint_iteration_prefers_reference_names_and_keeps_legacy_support() -> None:
  assert checkpoint_iteration("checkpoint_42.pt") == 42
  assert checkpoint_iteration(Path("checkpoint_final.pt")) is None
  assert checkpoint_iteration("model_42.pt") == 42
  assert checkpoint_iteration(Path("model_final.pt")) is None


def test_checkpoint_sort_key_keeps_checkpoint_final_last() -> None:
  names = [
    "checkpoint_9.pt",
    "checkpoint_final.pt",
    "checkpoint_100.pt",
    "checkpoint_10.pt",
  ]

  assert sorted(names, key=checkpoint_sort_key) == [
    "checkpoint_9.pt",
    "checkpoint_10.pt",
    "checkpoint_100.pt",
    "checkpoint_final.pt",
  ]


def test_resolve_local_checkpoint_path_selects_latest_source_style_checkpoint(tmp_path: Path) -> None:
  log_root = tmp_path / "logs" / "rsl_rl" / "g1_tracking"
  first = log_root / "2026-07-09_10-00-00_a"
  second = log_root / "2026-07-09_11-00-00_b"
  first.mkdir(parents=True)
  second.mkdir(parents=True)
  (first / "checkpoint_100.pt").write_text("old")
  (second / "checkpoint_9.pt").write_text("new")
  (second / "checkpoint_final.pt").write_text("final")

  resolved = resolve_local_checkpoint_path(
    log_root=log_root,
    load_run=".*",
    load_checkpoint="checkpoint_.*.pt",
  )

  assert resolved == second / "checkpoint_final.pt"


def test_resolve_local_checkpoint_path_rejects_missing_match(tmp_path: Path) -> None:
  log_root = tmp_path / "logs" / "rsl_rl" / "g1_tracking"
  (log_root / "run").mkdir(parents=True)

  with pytest.raises(ValueError, match="No checkpoint found"):
    resolve_local_checkpoint_path(
      log_root=log_root,
      load_run=".*",
      load_checkpoint="checkpoint_.*.pt",
    )
