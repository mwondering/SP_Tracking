import torch
from rsl_rl.algorithms import PPO

from sp_tracking.tasks.tracking.rl.ppo import SparseTrackSplitLrPPO


def _make_algorithm(actor_lr: float = 1.0e-4, critic_lr: float = 5.0e-4):
  algorithm = object.__new__(SparseTrackSplitLrPPO)
  algorithm.actor_learning_rate = actor_lr
  algorithm.critic_learning_rate = critic_lr
  algorithm.learning_rate = actor_lr
  actor_param = torch.nn.Parameter(torch.zeros(()))
  critic_param = torch.nn.Parameter(torch.zeros(()))
  algorithm.optimizer = torch.optim.Adam(
    [
      {"params": [actor_param], "lr": actor_lr},
      {"params": [critic_param], "lr": critic_lr},
    ]
  )
  return algorithm


def test_split_lr_state_round_trips_through_ppo_save_and_load(monkeypatch) -> None:
  source = _make_algorithm(actor_lr=2.5e-5, critic_lr=1.25e-4)
  monkeypatch.setattr(PPO, "save", lambda self: {"base": "state"})
  saved = source.save()

  assert saved["tracking_split_lr_state"] == {
    "actor_learning_rate": 2.5e-5,
    "critic_learning_rate": 1.25e-4,
    "learning_rate": 2.5e-5,
  }

  target = _make_algorithm()
  monkeypatch.setattr(PPO, "load", lambda self, loaded, cfg, strict: True)
  assert target.load(saved, load_cfg=None, strict=True) is True

  assert target.actor_learning_rate == 2.5e-5
  assert target.critic_learning_rate == 1.25e-4
  assert target.learning_rate == 2.5e-5
  assert target.optimizer.param_groups[0]["lr"] == 2.5e-5
  assert target.optimizer.param_groups[1]["lr"] == 1.25e-4


def test_split_lr_load_derives_state_from_legacy_optimizer_groups() -> None:
  algorithm = _make_algorithm(actor_lr=1.0e-4, critic_lr=5.0e-4)
  algorithm.optimizer.param_groups[0]["lr"] = 3.0e-5
  algorithm.optimizer.param_groups[1]["lr"] = 1.5e-4

  algorithm._restore_split_lr_checkpoint_state({}, load_cfg=None)

  assert algorithm.actor_learning_rate == 3.0e-5
  assert algorithm.critic_learning_rate == 1.5e-4
  assert algorithm.learning_rate == 3.0e-5


def test_split_lr_actor_only_load_does_not_change_optimizer_state() -> None:
  algorithm = _make_algorithm()
  algorithm._restore_split_lr_checkpoint_state(
    {
      "tracking_split_lr_state": {
        "actor_learning_rate": 2.5e-5,
        "critic_learning_rate": 1.25e-4,
        "learning_rate": 2.5e-5,
      }
    },
    load_cfg={"actor": True},
  )

  assert algorithm.actor_learning_rate == 1.0e-4
  assert algorithm.critic_learning_rate == 5.0e-4
  assert algorithm.optimizer.param_groups[0]["lr"] == 1.0e-4
  assert algorithm.optimizer.param_groups[1]["lr"] == 5.0e-4
