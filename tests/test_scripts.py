from pathlib import Path


def test_tracking_bfm_training_script_contract() -> None:
  root = Path(__file__).resolve().parents[1]
  script = root / "scripts" / "train_tracking_bfm.sh"

  assert script.exists()
  assert script.stat().st_mode & 0o111

  contents = script.read_text()
  assert "uv run sp-train" in contents
  assert "task=tracking_bfm" in contents
  assert "launch_script_path=" in contents
  assert "agent.max_iterations=" in contents
  assert "agent.logger=wandb" in contents
  assert "agent.upload_model=False" in contents
  assert '"$@"' in contents
  assert "smoke" not in script.name
  assert not (root / "scripts" / "train_tracking_bfm_smoke.sh").exists()
