from pathlib import Path

from sp_tracking.scripts.play import PlayConfig, prepare_play_cfg


def test_prepare_play_cfg_applies_tracking_motion_and_checkpoint(tmp_path: Path) -> None:
  motion_file = tmp_path / "motion.npz"
  checkpoint_file = tmp_path / "model_100.pt"
  motion_file.write_bytes(b"motion")
  checkpoint_file.write_bytes(b"ckpt")

  prepared = prepare_play_cfg(
    PlayConfig(
      task="tracking_bfm",
      checkpoint_file=str(checkpoint_file),
      motion_file=str(motion_file),
      num_envs=3,
      domain_randomization=False,
    )
  )

  assert prepared.checkpoint_path == checkpoint_file
  assert prepared.env.scene.num_envs == 3
  assert prepared.env.commands["motion"].motion_file == str(motion_file)
  assert prepared.env.events == {}


def test_prepare_play_cfg_accepts_motion_path_for_largedataset(tmp_path: Path) -> None:
  motion_path = tmp_path / "motions"
  checkpoint_file = tmp_path / "model_final.pt"
  motion_path.mkdir()
  checkpoint_file.write_bytes(b"ckpt")

  prepared = prepare_play_cfg(
    PlayConfig(
      task="tracking_bfm_largedataset",
      checkpoint_file=str(checkpoint_file),
      motion_path=str(motion_path),
    )
  )

  assert prepared.env.commands["motion"].motion_path == str(motion_path)
