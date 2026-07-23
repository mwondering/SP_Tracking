import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_module
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from sp_tracking.config.build_agent import build_agent_cfg, serialize_agent_cfg
from sp_tracking.tasks.tracking.rl.export import export_sim2real_policy_onnx
from sp_tracking.tasks.tracking.rl.ppo import SparseTrackSplitLrPPO
from sp_tracking.tasks.tracking.rl.sapg.conditioning import (
  BlockGaussianDistribution,
  PolicyConditionedLinear,
  PolicyContext,
)
from sp_tracking.tasks.tracking.rl.sapg.config import SAPGConfig
from sp_tracking.tasks.tracking.rl.sapg.extension import SAPGRuntime
from sp_tracking.tasks.tracking.task_catalog import TASK_SPECS


class _FakeEnv:
  num_envs = 8
  num_actions = 2


def _observations(value: float = 0.0, num_envs: int = 8) -> TensorDict:
  return TensorDict(
    {
      "actor_obs": torch.full((num_envs, 3), value),
      "critic_obs": torch.full((num_envs, 4), value),
    },
    batch_size=[num_envs],
  )


def _config() -> dict:
  return {
    "actor": {
      "class_name": "MLPModel",
      "hidden_dims": [16, 8],
      "activation": "elu",
      "obs_normalization": False,
      "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 0.7,
        "std_type": "scalar",
      },
    },
    "critic": {
      "class_name": "MLPModel",
      "hidden_dims": [16, 8],
      "activation": "elu",
      "obs_normalization": False,
      "distribution_cfg": None,
    },
    "algorithm": {
      "class_name": "sp_tracking.tasks.tracking.rl.ppo:SparseTrackSplitLrPPO",
      "value_loss_coef": 1.0,
      "use_clipped_value_loss": True,
      "clip_param": 0.2,
      "entropy_coef": 0.0,
      "num_learning_epochs": 1,
      "num_mini_batches": 2,
      "learning_rate": 1.0e-3,
      "schedule": "fixed",
      "gamma": 0.99,
      "lam": 0.95,
      "desired_kl": None,
      "max_grad_norm": 1.0,
      "sapg_cfg": {
        "enabled": True,
        "compatibility": "official",
        "num_policy_blocks": 4,
        "local_parameter_dim": 5,
        "off_policy_ratio": 1,
        "exploration_type": "none",
        "entropy_coef_scale": 1.0,
        "value_eval_chunk_size": 7,
      },
    },
    "obs_groups": {
      "actor": ["actor_obs"],
      "critic": ["critic_obs"],
    },
    "num_steps_per_env": 3,
    "multi_gpu": None,
    "torch_compile_mode": None,
  }


def test_plain_ppo_runs_one_sapg_rollout_and_update() -> None:
  algorithm = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(), _FakeEnv(), deepcopy(_config()), "cpu"
  )
  assert isinstance(algorithm.actor.mlp[0], PolicyConditionedLinear)
  assert isinstance(algorithm.actor.distribution, BlockGaussianDistribution)
  auxiliary_sample_counts = []

  def auxiliary_loss(observations):
    auxiliary_sample_counts.append(observations.batch_size[0])
    zero = algorithm.actor.mlp[0].policy_weight.sum() * 0.0
    return zero, {"test_auxiliary": zero}

  algorithm._auxiliary_loss = auxiliary_loss

  obs = _observations()
  for step in range(3):
    actions = algorithm.act(obs)
    assert actions.shape == (8, 2)
    next_obs = _observations(float(step + 1))
    algorithm.process_env_step(
      next_obs,
      torch.ones(8),
      torch.zeros(8, dtype=torch.bool),
      {},
    )
    obs = next_obs
  algorithm.compute_returns(obs)
  losses = algorithm.update()

  assert algorithm.storage.step == 0
  assert losses["sapg/off_policy_fraction"] == pytest.approx(0.2)
  # Base mini-batch is 12 samples; official batching folds the six-sample
  # remainder into the second batch instead of adding an optimizer step.
  assert losses["sapg/num_updates"] == 2.0
  assert sum(auxiliary_sample_counts) == 24


def test_training_act_rejects_non_divisible_environment_count_immediately() -> None:
  class InvalidEnv:
    num_envs = 5
    num_actions = 2

  algorithm = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(num_envs=5), InvalidEnv(), deepcopy(_config()), "cpu"
  )

  with pytest.raises(
    ValueError, match=r"requires num_envs \(5\) to be divisible"
  ):
    algorithm.act(_observations(num_envs=5))


def test_off_policy_clip_fraction_is_not_divided_by_epochs_twice() -> None:
  cfg = _config()
  cfg["algorithm"]["num_learning_epochs"] = 3
  algorithm = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(), _FakeEnv(), deepcopy(cfg), "cpu"
  )
  obs = _observations()
  for step in range(3):
    algorithm.act(obs)
    next_obs = _observations(float(step + 1))
    algorithm.process_env_step(
      next_obs,
      torch.ones(8),
      torch.zeros(8, dtype=torch.bool),
      {},
    )
    obs = next_obs
  algorithm.compute_returns(obs)
  with torch.no_grad():
    algorithm.actor.mlp[-1].bias.add_(100.0)
  losses = algorithm.update()

  assert losses["sapg/off_policy_clip_fraction"] == pytest.approx(1.0)


def test_adaptive_kl_uses_refreshed_distribution_in_next_epoch(
  monkeypatch,
) -> None:
  cfg = _config()
  cfg["algorithm"].update(
    {
      "num_learning_epochs": 2,
      "num_mini_batches": 1,
      "learning_rate": 0.0,
      "schedule": "adaptive",
      "desired_kl": 0.01,
    }
  )
  algorithm = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(), _FakeEnv(), deepcopy(cfg), "cpu"
  )
  conditioned = algorithm.actor.mlp[0]
  with torch.no_grad():
    conditioned.policy_embedding.zero_()
    conditioned.policy_embedding[-1].fill_(1.0)
    conditioned.policy_weight.fill_(0.25)

  obs = _observations()
  for step in range(3):
    algorithm.act(obs)
    next_obs = _observations(float(step + 1))
    algorithm.process_env_step(
      next_obs,
      torch.ones(8),
      torch.zeros(8, dtype=torch.bool),
      {},
    )
    obs = next_obs
  algorithm.compute_returns(obs)

  kl_values: list[float] = []

  def record_kl(_algorithm, kl_mean):
    kl_values.append(float(kl_mean.item()))

  monkeypatch.setattr(
    "sp_tracking.tasks.tracking.rl.sapg.update._update_learning_rate",
    record_kl,
  )
  algorithm.update()

  assert len(kl_values) == 2
  assert kl_values[0] > 0.0
  assert kl_values[1] == pytest.approx(0.0, abs=1.0e-8)


def test_disabled_sparse_constructor_delegates_to_upstream_ppo(monkeypatch) -> None:
  sentinel = object()
  calls = []

  def construct(obs, env, cfg, device):
    calls.append((obs, env, cfg, device))
    return sentinel

  monkeypatch.setattr(PPO, "construct_algorithm", construct)
  cfg = {"algorithm": {}}
  assert SparseTrackSplitLrPPO.construct_algorithm(
    "obs", "env", cfg, "cpu"
  ) is sentinel
  assert calls == [("obs", "env", cfg, "cpu")]


def test_nonzero_rank_uses_rank_zero_follower_selection(monkeypatch) -> None:
  algorithm = SimpleNamespace(
    gpu_global_rank=1,
    is_multi_gpu=True,
    device="cpu",
  )
  runtime = SAPGRuntime(
    algorithm,
    SAPGConfig(enabled=True, off_policy_ratio=2),
    PolicyContext(4),
  )

  def broadcast(selected, src):
    assert src == 0
    selected.copy_(torch.tensor([2, 0]))

  monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
  assert runtime._select_followers().tolist() == [2, 0]


def _distributed_sapg_worker(rank: int, rendezvous_path: str) -> None:
  os.environ["GLOO_SOCKET_IFNAME"] = "lo"
  torch.distributed.init_process_group(
    "gloo",
    init_method=f"file://{rendezvous_path}",
    rank=rank,
    world_size=2,
  )
  try:
    cfg = _config()
    cfg["multi_gpu"] = {"global_rank": rank, "world_size": 2}
    algorithm = SparseTrackSplitLrPPO.construct_algorithm(
      _observations(), _FakeEnv(), cfg, "cpu"
    )
    algorithm.broadcast_parameters()
    obs = _observations(float(rank))
    for step in range(3):
      algorithm.act(obs)
      next_obs = _observations(float(rank + step + 1))
      algorithm.process_env_step(
        next_obs,
        torch.ones(8),
        torch.zeros(8, dtype=torch.bool),
        {},
      )
      obs = next_obs
    algorithm.compute_returns(obs)
    algorithm.update()

    follower = algorithm._sapg_runtime.last_selected_follower_ids
    gathered_followers = [torch.empty_like(follower) for _ in range(2)]
    torch.distributed.all_gather(gathered_followers, follower)
    assert torch.equal(gathered_followers[0], gathered_followers[1])

    parameters = torch.cat(
      [
        parameter.detach().reshape(-1)
        for model in (algorithm.actor, algorithm.critic)
        for parameter in model.parameters()
      ]
    )
    gathered_parameters = [torch.empty_like(parameters) for _ in range(2)]
    torch.distributed.all_gather(gathered_parameters, parameters)
    torch.testing.assert_close(gathered_parameters[0], gathered_parameters[1])
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
  os.environ.get("SP_TRACKING_RUN_DISTRIBUTED_TESTS") != "1",
  reason="requires local Gloo sockets, which are unavailable in some sandboxes",
)
def test_two_rank_gloo_synchronizes_follower_and_gradients(tmp_path) -> None:
  torch.multiprocessing.start_processes(
    _distributed_sapg_worker,
    args=(str(tmp_path / "sapg_gloo_init"),),
    nprocs=2,
    join=True,
    start_method="fork",
  )


def test_direct_inference_without_context_uses_leader() -> None:
  class SingleEnv:
    num_envs = 1
    num_actions = 2

  algorithm = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(num_envs=1), SingleEnv(), deepcopy(_config()), "cpu"
  )
  context = algorithm._sapg_runtime.context
  obs = _observations(num_envs=1)

  direct = algorithm.get_policy()(obs)
  with context.use(torch.full((1,), 3, dtype=torch.long)):
    explicit_leader = algorithm.get_policy()(obs)
  torch.testing.assert_close(direct, explicit_leader)


def test_sapg_checkpoint_strictly_round_trips_runtime_and_models() -> None:
  source = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(), _FakeEnv(), deepcopy(_config()), "cpu"
  )
  source._sapg_runtime._select_followers()
  saved = source.save()
  target = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(), _FakeEnv(), deepcopy(_config()), "cpu"
  )

  assert target.load(saved, load_cfg=None, strict=True) is True
  assert target._sapg_runtime.save()["config"] == saved["tracking_sapg_state"][
    "config"
  ]
  for source_parameter, target_parameter in zip(
    source.actor.parameters(), target.actor.parameters(), strict=True
  ):
    torch.testing.assert_close(source_parameter, target_parameter)


def test_base_ppo_actor_critic_can_warm_start_sapg_without_shape_errors() -> None:
  sapg = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(), _FakeEnv(), deepcopy(_config()), "cpu"
  )
  base_actor_state = {
    name: value.clone()
    for name, value in sapg.actor.state_dict().items()
    if "policy_weight" not in name and "policy_embedding" not in name
  }
  base_actor_state["distribution.std_param"] = base_actor_state[
    "distribution.std_param"
  ][0].clone()
  base_critic_state = {
    name: value.clone()
    for name, value in sapg.critic.state_dict().items()
    if "policy_weight" not in name
  }
  checkpoint = {
    "actor_state_dict": base_actor_state,
    "critic_state_dict": base_critic_state,
  }

  assert sapg.load(
    checkpoint,
    load_cfg={"actor": True, "critic": True, "optimizer": False},
    strict=False,
  ) is False
  expected_std = base_actor_state["distribution.std_param"].repeat(4, 1)
  torch.testing.assert_close(sapg.actor.distribution.std_param, expected_std)


def test_jit_export_wrapper_matches_live_leader_policy() -> None:
  algorithm = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(), _FakeEnv(), deepcopy(_config()), "cpu"
  )
  policy = algorithm.get_policy()
  obs = _observations(0.25)

  expected = policy(obs)
  exported = policy.as_jit()
  assert isinstance(exported.mlp[0], torch.nn.Linear)
  scripted = torch.jit.script(exported)
  actual = exported(obs["actor_obs"])
  torch.testing.assert_close(actual, expected)
  torch.testing.assert_close(scripted(obs["actor_obs"]), expected)


def test_onnx_export_keeps_original_input_contract(tmp_path) -> None:
  algorithm = SparseTrackSplitLrPPO.construct_algorithm(
    _observations(), _FakeEnv(), deepcopy(_config()), "cpu"
  )
  output_path = tmp_path / "sapg_policy.onnx"

  export_sim2real_policy_onnx(
    policy=algorithm.get_policy(),
    env=_FakeEnv(),
    path=output_path,
    run_name="sapg-test",
    iteration=0,
    checkpoint_name="checkpoint.pt",
  )

  assert output_path.is_file()
  assert output_path.with_suffix(".json").is_file()


@pytest.mark.parametrize(
  "task_spec",
  [spec for spec in TASK_SPECS if not spec.is_experiment],
  ids=lambda spec: spec.config_name,
)
def test_every_catalog_task_accepts_the_same_sapg_switch(task_spec) -> None:
  overrides = [
    *task_spec.hydra_overrides,
    "agent.algorithm.sapg_cfg.enabled=true",
  ]
  with initialize_config_module(
    version_base=None, config_module="sp_tracking.conf"
  ):
    cfg = compose(config_name="train", overrides=overrides)
  agent_cfg = build_agent_cfg(cfg.agent, cfg.task.get("agent_overrides"))
  serialized = serialize_agent_cfg(agent_cfg)

  assert serialized["algorithm"]["sapg_cfg"]["enabled"] is True
  assert serialized["algorithm"]["class_name"] != "PPO"
