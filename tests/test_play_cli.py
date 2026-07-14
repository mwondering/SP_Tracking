from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_module

from sp_tracking.scripts.play import PlayConfig, prepare_play_cfg


def _save_local_checkpoint(
  tmp_path: Path, task: str, *overrides: str
) -> Path:
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    cfg = compose(
      config_name="train",
      overrides=[f"task={task}", *overrides],
  )
  checkpoint = tmp_path / "checkpoint_final.pt"
  torch.save({"cfg": cfg}, checkpoint)
  return checkpoint


def test_prepare_play_cfg_uses_embedded_tracking_bfm_cfg(tmp_path: Path) -> None:
  motion_file = tmp_path / "motion.npz"
  motion_file.write_bytes(b"motion")
  checkpoint_file = _save_local_checkpoint(
    tmp_path, "tracking_bfm", "motion_path=/dataset/from_training"
  )

  prepared = prepare_play_cfg(
    PlayConfig(
      checkpoint_file=str(checkpoint_file),
      motion_file=str(motion_file),
      num_envs=3,
      domain_randomization=False,
    )
  )

  assert prepared.checkpoint_path == checkpoint_file
  assert prepared.env.scene.num_envs == 3
  assert prepared.env.commands["motion"].motion_file == str(motion_file)
  assert prepared.env.commands["motion"].motion_path == ""
  assert prepared.env.events == {}
  assert prepared.agent.actor.hidden_dims == (2048, 2048, 1024, 1024, 512, 256, 128)


def test_prepare_play_cfg_uses_embedded_sp_cfg_without_task_flag(tmp_path: Path) -> None:
  checkpoint_file = _save_local_checkpoint(tmp_path, "tracking_bfm_sp")
  motion_path = tmp_path / "motions"
  motion_path.mkdir()

  prepared = prepare_play_cfg(
    PlayConfig(checkpoint_file=str(checkpoint_file), motion_path=str(motion_path))
  )

  assert prepared.env.commands["motion"].motion_path == str(motion_path)
  assert list(prepared.env.observations) == ["policy", "priv", "priv_critic"]
  assert prepared.agent.obs_groups == {
    "actor": ("policy", "priv"),
    "critic": ("policy", "priv", "priv_critic"),
  }


def test_prepare_play_cfg_requires_task_for_legacy_local_checkpoint(tmp_path: Path) -> None:
  checkpoint_file = tmp_path / "checkpoint_final.pt"
  torch.save({}, checkpoint_file)

  with pytest.raises(ValueError, match="no embedded cfg"):
    prepare_play_cfg(PlayConfig(checkpoint_file=str(checkpoint_file)))

  prepared = prepare_play_cfg(
    PlayConfig(task="tracking_bfm", checkpoint_file=str(checkpoint_file))
  )
  assert list(prepared.env.observations) == ["actor", "critic"]


@pytest.mark.parametrize(
  ("task", "obs_groups"),
  [
    (
      "tracking_bfm_sp_ablation_bfm_actor",
      ["actor", "policy", "priv"],
    ),
    (
      "tracking_bfm_sp_ablation_student_actor",
      ["actor", "policy", "priv"],
    ),
    (
      "tracking_bfm_sp_ablation_teacher_actor",
      ["actor", "policy", "priv"],
    ),
  ],
)
def test_prepare_play_cfg_supports_new_variant_for_legacy_checkpoint(
  tmp_path: Path, task: str, obs_groups: list[str]
) -> None:
  checkpoint_file = tmp_path / "checkpoint_final.pt"
  torch.save({}, checkpoint_file)

  prepared = prepare_play_cfg(
    PlayConfig(task=task, checkpoint_file=str(checkpoint_file))  # type: ignore[arg-type]
  )

  assert list(prepared.env.observations) == obs_groups


def test_prepare_play_cfg_rejects_task_mismatch_with_local_checkpoint(tmp_path: Path) -> None:
  checkpoint_file = _save_local_checkpoint(tmp_path, "tracking_bfm_sp")

  with pytest.raises(ValueError, match="does not match checkpoint task"):
    prepare_play_cfg(
      PlayConfig(task="tracking_bfm", checkpoint_file=str(checkpoint_file))
    )


def test_prepare_play_cfg_rejects_both_motion_sources(tmp_path: Path) -> None:
  checkpoint_file = _save_local_checkpoint(tmp_path, "tracking_bfm_sp")
  motion_file = tmp_path / "motion.npz"
  motion_file.write_bytes(b"motion")
  motion_path = tmp_path / "motions"
  motion_path.mkdir()

  with pytest.raises(ValueError, match="either motion_file or motion_path"):
    prepare_play_cfg(
      PlayConfig(
        checkpoint_file=str(checkpoint_file),
        motion_file=str(motion_file),
        motion_path=str(motion_path),
      )
    )
