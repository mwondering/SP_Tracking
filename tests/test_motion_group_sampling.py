import torch

from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  _build_motion_group_probabilities,
)


GROUPS = [
  {"path_fragment": "lafan", "weight": 0.2},
  {"path_fragment": "100style", "weight": 0.3},
  {"path_fragment": "seed/all", "weight": 0.3},
]


def test_motion_group_probabilities_are_uniform_within_weighted_groups() -> None:
  probabilities = _build_motion_group_probabilities(
    [
      "/data/g1/lafan/a.npz",
      "/data/g1/lafan/b.npz",
      "/data/g1/100style/c.npz",
      "/data/g1/seed/all/d.npz",
    ],
    GROUPS,
    "cpu",
  )

  assert probabilities is not None
  torch.testing.assert_close(
    probabilities, torch.tensor([0.125, 0.125, 0.375, 0.375])
  )


def test_motion_group_probabilities_fall_back_for_custom_flat_dataset() -> None:
  probabilities = _build_motion_group_probabilities(
    ["/custom/a.npz", "/custom/b.npz"], GROUPS, "cpu"
  )

  assert probabilities is None
