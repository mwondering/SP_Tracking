from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.rl.runner import (
  _FirstNonfiniteSimulationTracer,
  _format_nonfinite_action_diagnostics,
  _format_nonfinite_env_output_diagnostics,
  _format_nonfinite_internal_state_diagnostics,
)


class _CommandManager:
  def __init__(self, command):
    self.command = command

  def get_term(self, name: str):
    assert name == "motion"
    return self.command


def test_nan_diagnostics_identify_concatenated_obs_term_and_motion_file() -> None:
  obs = {
    "actor": torch.tensor(
      [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, float("nan"), 5.0],
      ]
    )
  }
  command = SimpleNamespace(
    motion_idx=torch.tensor([0, 3]),
    time_steps=torch.tensor([11, 17]),
    motion_store=SimpleNamespace(
      motion_files=["m0.npz", "m1.npz", "m2.npz", "bad_motion.npz"]
    ),
  )
  unwrapped = SimpleNamespace(
    observation_manager=SimpleNamespace(
      active_terms={"actor": ["good", "bad"]},
      group_obs_term_dim={"actor": [(2,), (3,)]},
      group_obs_concatenate={"actor": True},
    ),
    command_manager=_CommandManager(command),
  )
  env = SimpleNamespace(unwrapped=unwrapped)

  message = _format_nonfinite_env_output_diagnostics(
    env,
    obs,
    rewards=torch.zeros(2),
    dones=torch.zeros(2, dtype=torch.long),
  )

  assert "obs[actor]/bad" in message
  assert "envs=[1]" in message
  assert "motion_ids=[3]" in message
  assert "time_steps=[17]" in message
  assert "bad_motion.npz" in message


def test_action_diagnostics_identify_nonfinite_policy_actions() -> None:
  command = SimpleNamespace(
    motion_idx=torch.tensor([2, 4]),
    time_steps=torch.tensor([5, 9]),
    motion_store=SimpleNamespace(
      motion_files=["m0.npz", "m1.npz", "source.npz", "m3.npz", "nan_action.npz"]
    ),
  )
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(command_manager=_CommandManager(command))
  )

  message = _format_nonfinite_action_diagnostics(
    env,
    torch.tensor([[0.0, 1.0], [float("nan"), 0.5]]),
  )

  assert "Policy action contains NaN/Inf" in message
  assert "envs=[1]" in message
  assert "motion_ids=[4]" in message
  assert "time_steps=[9]" in message
  assert "nan_action.npz" in message


def test_internal_diagnostics_identify_exact_sim_field_and_index() -> None:
  data = SimpleNamespace(
    qpos=torch.tensor([[0.0, 1.0], [0.0, float("nan")]]),
    qvel=torch.zeros((2, 2)),
    qacc=torch.zeros((2, 2)),
    qacc_warmstart=torch.zeros((2, 2)),
    ctrl=torch.zeros((2, 1)),
    nefc=torch.tensor([12, 2186]),
  )
  command = SimpleNamespace(
    motion_idx=torch.tensor([0, 3]),
    time_steps=torch.tensor([11, 17]),
    motion_store=SimpleNamespace(
      motion_files=["m0.npz", "m1.npz", "m2.npz", "bad_motion.npz"]
    ),
  )
  unwrapped = SimpleNamespace(
    sim=SimpleNamespace(data=data, mj_model=None),
    action_manager=None,
    command_manager=_CommandManager(command),
  )
  env = SimpleNamespace(unwrapped=unwrapped)

  message = _format_nonfinite_internal_state_diagnostics(env, [1])

  assert "sim.data.qpos" in message
  assert "env=1 index=(1,)" in message
  assert "nefc(bad_envs)={1: [2186]}" in message
  assert "bad_motion.npz" in message


def test_first_nonfinite_tracer_stops_on_first_bad_physics_substep() -> None:
  data = SimpleNamespace(
    qpos=torch.zeros((2, 2)),
    qvel=torch.zeros((2, 2)),
    qacc=torch.zeros((2, 2)),
    qacc_warmstart=torch.zeros((2, 2)),
    ctrl=torch.zeros((2, 1)),
    sensordata=torch.zeros((2, 1)),
  )

  def sim_step() -> None:
    data.qacc[1, 0] = float("nan")

  sim = SimpleNamespace(data=data, mj_model=None, step=sim_step)
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      sim=sim,
      action_manager=None,
      command_manager=None,
    )
  )
  tracer = _FirstNonfiniteSimulationTracer(env)
  tracer.begin_env_step(iteration=7, rollout_step=3)

  try:
    sim.step()
  except FloatingPointError as exc:
    message = str(exc)
  else:
    raise AssertionError("Tracer should stop on a non-finite simulator state")

  assert "FIRST_NONFINITE_STATE" in message
  assert "iteration=7 rollout_step=3 physics_substep=1" in message
  assert "sim.data.qacc" in message
  assert "env=1 index=(0,)" in message


def test_first_nonfinite_tracer_distinguishes_bad_solver_input() -> None:
  data = SimpleNamespace(
    qpos=torch.zeros((1, 2)),
    qvel=torch.zeros((1, 2)),
    qacc=torch.zeros((1, 2)),
    qacc_warmstart=torch.zeros((1, 2)),
    ctrl=torch.zeros((1, 1)),
    qfrc_applied=torch.zeros((1, 2)),
    xfrc_applied=torch.zeros((1, 1, 6)),
    sensordata=torch.zeros((1, 1)),
  )
  sim_calls = 0

  def sim_step() -> None:
    nonlocal sim_calls
    sim_calls += 1

  sim = SimpleNamespace(data=data, mj_model=None, step=sim_step)
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      sim=sim,
      action_manager=None,
      command_manager=None,
    )
  )
  tracer = _FirstNonfiniteSimulationTracer(env)
  tracer.begin_env_step(iteration=2, rollout_step=4)
  data.ctrl[0, 0] = float("nan")

  try:
    sim.step()
  except FloatingPointError as exc:
    message = str(exc)
  else:
    raise AssertionError("Tracer should reject a non-finite solver input")

  assert "phase=before sim.step" in message
  assert "sim.data.ctrl" in message
  assert sim_calls == 0
