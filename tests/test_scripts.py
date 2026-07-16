from pathlib import Path
import subprocess

def test_tracking_bfm_training_script_contract() -> None:
  root = Path(__file__).resolve().parents[1]
  script = root / "scripts" / "train_tracking_bfm.sh"

  assert script.exists()
  assert script.stat().st_mode & 0o111

  contents = script.read_text()
  assert "uv run sp-train" in contents
  assert 'TASK_ID="${SP_TRACKING_TASK_ID:-SPTracking-G1-BFM-BFMActor-BFMCritic}"' in contents
  assert 'cmd+=("task_id=${TASK_ID}")' in contents
  assert "launch_script_path=" in contents
  assert "agent.max_iterations=" in contents
  assert "agent.logger=wandb" in contents
  assert "agent.upload_model=False" in contents
  assert '"$@"' in contents
  assert "smoke" not in script.name
  assert not (root / "scripts" / "train_tracking_bfm_smoke.sh").exists()


def test_tracking_bfm_multigpu_script_uses_torchrun() -> None:
  root = Path(__file__).resolve().parents[1]
  script = root / "scripts" / "train_tracking_bfm_multigpu.sh"

  assert script.exists()
  assert script.stat().st_mode & 0o111

  contents = script.read_text()
  assert "uv run torchrun" in contents
  assert "--standalone" in contents
  assert "--nproc_per_node" in contents
  assert "-m sp_tracking.scripts.train" in contents
  assert "CUDA_VISIBLE_DEVICES" in contents
  assert 'TASK_ID="${SP_TRACKING_TASK_ID:-SPTracking-G1-BFM-BFMActor-BFMCritic}"' in contents
  assert 'cmd+=("task_id=${TASK_ID}")' in contents
  assert "launch_script_path=" in contents
  assert '"$@"' in contents


def test_tracking_bfm_play_script_contract() -> None:
  root = Path(__file__).resolve().parents[1]
  script = root / "scripts" / "play_tracking_bfm.sh"

  assert script.exists()
  assert script.stat().st_mode & 0o111

  contents = script.read_text()
  assert "uv run sp-play" in contents
  assert "--checkpoint-file" in contents
  assert "--wandb-run-path" not in contents
  assert "--wandb-checkpoint-name" not in contents
  assert "--motion-file" in contents
  assert "--motion-path" in contents
  assert "--task-id" in contents
  assert "--dry-run" in contents
  assert "tracking_bfm_largedataset" not in contents
  # The launcher delegates validation to the canonical Python catalog instead
  # of maintaining a second hard-coded task list.
  assert "tracking_bfm_spv5_actor_heft_critic_heft_reward" not in contents


def test_play_script_accepts_canonical_public_task_id(tmp_path: Path) -> None:
  root = Path(__file__).resolve().parents[1]
  script = root / "scripts" / "play_tracking_bfm.sh"
  checkpoint = tmp_path / "checkpoint_final.pt"
  checkpoint.write_bytes(b"dry-run")
  task_id = "SPTracking-G1-BFM-SPV5Actor-HEFTCritic-HEFTReward"

  completed = subprocess.run(
    (
      str(script),
      "--task-id",
      task_id,
      "--checkpoint-file",
      str(checkpoint),
      "--dry-run",
    ),
    check=True,
    capture_output=True,
    text=True,
  )

  assert f"--task-id {task_id}" in completed.stdout
