import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from sp_tracking.tasks.tracking.rl.sapg.batch import (
  build_aggregated_data,
  rollout_policy_ids,
  sapg_mini_batch_generator,
)


def _storage() -> RolloutStorage:
  obs = TensorDict(
    {"actor": torch.zeros(4, 1)}, batch_size=[4]
  )
  storage = RolloutStorage("rl", 4, 3, obs, [1], "cpu")
  storage.observations["actor"] = torch.arange(12).reshape(3, 4, 1).float()
  storage.actions = storage.observations["actor"].clone()
  storage.values = torch.arange(12).reshape(3, 4, 1).float()
  storage.returns = storage.values + 10.0
  storage.rewards.fill_(1.0)
  storage.dones.zero_()
  storage.actions_log_prob.zero_()
  storage.distribution_params = (
    torch.zeros(3, 4, 1),
    torch.ones(3, 4, 1),
  )
  return storage


def test_rollout_policy_ids_are_contiguous_and_leader_is_last() -> None:
  assert rollout_policy_ids(8, 4, "cpu").tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
  assert rollout_policy_ids(
    1, 4, "cpu", require_divisible=False
  ).tolist() == [3]


def test_aggregate_keeps_all_on_policy_and_maps_selected_follower_to_leader() -> None:
  storage = _storage()
  leader_values = torch.full((3, 1, 1), 20.0)
  leader_next_values = torch.full((3, 1, 1), 30.0)
  data = build_aggregated_data(
    storage,
    torch.tensor([1]),
    leader_values,
    leader_next_values,
    num_policy_blocks=4,
    gamma=0.5,
  )

  assert data.num_samples == 15
  assert data.off_policy_mask.tolist() == [False] * 12 + [True] * 3
  assert data.source_policy_ids[-3:].tolist() == [1, 1, 1]
  assert data.target_policy_ids[-3:].tolist() == [3, 3, 3]
  # Follower block 1 is environment 1; its time-major flat indices are 1,5,9.
  assert data.source_indices[-3:].tolist() == [1, 5, 9]
  torch.testing.assert_close(data.values[-3:], torch.full((3, 1), 20.0))
  torch.testing.assert_close(data.returns[-3:], torch.full((3, 1), 16.0))


def test_aggregate_uses_post_action_done_for_one_step_target() -> None:
  storage = _storage()
  storage.dones[1, 0] = 1
  data = build_aggregated_data(
    storage,
    torch.tensor([0]),
    torch.zeros(3, 1, 1),
    torch.full((3, 1, 1), 4.0),
    num_policy_blocks=4,
    gamma=0.5,
  )
  torch.testing.assert_close(
    data.returns[-3:], torch.tensor([[3.0], [1.0], [3.0]])
  )


def test_generator_folds_remainder_into_last_batch_and_reuses_shuffle() -> None:
  storage = _storage()
  data = build_aggregated_data(
    storage,
    torch.tensor([0]),
    torch.zeros(3, 1, 1),
    torch.zeros(3, 1, 1),
    num_policy_blocks=4,
    gamma=0.5,
  )
  batches = list(
    sapg_mini_batch_generator(
      storage, data, num_mini_batches=2, num_epochs=2
    )
  )
  assert [batch.observations.batch_size[0] for batch in batches] == [6, 9, 6, 9]
  first_epoch = torch.cat([batch.actions for batch in batches[:2]])
  second_epoch = torch.cat([batch.actions for batch in batches[2:]])
  torch.testing.assert_close(first_epoch, second_epoch)
