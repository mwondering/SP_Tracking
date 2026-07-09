from pathlib import Path


def test_tracking_bfm_scripts_match_old_h100_command_horizon_defaults() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  scripts = (
    repo_root / "scripts" / "train_tracking_bfm.sh",
    repo_root / "scripts" / "train_tracking_bfm_multigpu.sh",
  )

  for script in scripts:
    text = script.read_text()
    assert '"task.command.command.history_steps=${SP_TRACKING_HISTORY_STEPS:-0}"' in text
    assert '"task.command.command.future_steps=${SP_TRACKING_FUTURE_STEPS:-1}"' in text
