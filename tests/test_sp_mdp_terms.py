from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.mdp import randomizations as sp_randomizations
from sp_tracking.tasks.tracking.mdp import sp as sp_mdp


class _FakeActionManager:
  def __init__(self, applied_action: torch.Tensor, raw_action: torch.Tensor):
    self.action = raw_action
    self.prev_action = torch.zeros_like(raw_action)
    self._term = SimpleNamespace(applied_action=applied_action)

  def get_term(self, name: str):
    assert name == "joint_pos"
    return self._term


class _FakeEventManager:
  def __init__(self, terms: dict[str, object]):
    self._terms = terms

  def get_term_cfg(self, name: str):
    return SimpleNamespace(func=self._terms[name])


class _ObservedTerm:
  def __init__(self, value: torch.Tensor):
    self.value = value

  def observe(self, **_: object) -> torch.Tensor:
    return self.value


def test_spherical_noise_matches_source_bounded_radius_semantics() -> None:
  base = torch.zeros((4096, 3), dtype=torch.float32)
  noisy = sp_randomizations._add_spherical_noise(base, noise_std=0.1)
  radius = torch.linalg.vector_norm(noisy - base, dim=-1)

  assert torch.all(radius >= 0.0)
  assert torch.all(radius <= 0.1 + 1.0e-6)
  assert sp_randomizations._add_spherical_noise(base, noise_std=0.0) is base


def test_applied_action_reads_joint_position_action_term() -> None:
  applied = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
  raw = torch.zeros_like(applied)
  env = SimpleNamespace(action_manager=_FakeActionManager(applied, raw))

  assert sp_mdp.applied_action(env) is applied


def test_prev_actions_and_action_rate_use_action_term_history() -> None:
  history = torch.tensor([[[1.0, 2.0], [0.5, 1.0], [0.25, 0.5]]])

  class _HistoryTerm:
    applied_action = history[:, 0]

    def get_recent_action_obs(self, steps: int) -> torch.Tensor:
      return history[:, :steps]

    def get_recent_action_rate_actions(self, steps: int) -> torch.Tensor:
      return history[:, :steps]

  class _HistoryActionManager:
    action = history[:, 0]
    prev_action = torch.zeros_like(action)

    def get_term(self, name: str):
      assert name == "joint_pos"
      return _HistoryTerm()

  env = SimpleNamespace(num_envs=1, action_manager=_HistoryActionManager())

  assert torch.equal(sp_mdp.prev_actions(env, steps=3), history.reshape(1, -1))
  assert torch.equal(sp_mdp.action_rate_l2(env), -torch.tensor([1.25]))


def test_substep_cache_matches_source_joint_and_contact_aggregation() -> None:
  asset_data = SimpleNamespace(
    joint_pos=torch.zeros((1, 2)),
    joint_vel=torch.zeros((1, 2)),
  )
  sensor_data = SimpleNamespace(found=torch.zeros((1, 2)))
  sensor = SimpleNamespace(primary_names=("left", "right"), data=sensor_data)
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    physics_dt=0.005,
    common_step_counter=1,
    cfg=SimpleNamespace(decimation=4),
    scene={
      "robot": SimpleNamespace(joint_names=("j0", "j1"), data=asset_data),
      "contact_forces": sensor,
    },
  )
  cache = sp_mdp.substep_tracking_cache(SimpleNamespace(params={}), env)
  found_samples = (
    torch.tensor([[1.0, 0.0]]),
    torch.tensor([[1.0, 0.0]]),
    torch.tensor([[0.0, 0.0]]),
    torch.tensor([[0.0, 1.0]]),
  )
  for substep, found in enumerate(found_samples):
    asset_data.joint_pos[:] = torch.tensor([[float(substep), float(substep + 10)]])
    asset_data.joint_vel[:] = torch.tensor([[float(substep), float(substep * 2)]])
    sensor_data.found = found
    cache(env)

  assert torch.allclose(
    cache.joint_state_average("joint_pos"), torch.tensor([[2.5, 12.5]])
  )
  assert torch.allclose(
    cache.joint_state_average("joint_vel"), torch.tensor([[2.5, 5.0]])
  )
  current, first_contact, first_air = cache.contact_state()
  assert torch.equal(current, torch.tensor([[True, False]]))
  assert torch.equal(first_contact, torch.tensor([[True, False]]))
  assert torch.equal(first_air, torch.tensor([[False, False]]))
  assert torch.allclose(sp_mdp.joint_vel_l2(env), torch.tensor([-31.25]))


def test_domain_observations_read_randomization_terms() -> None:
  values = {
    "motor_params_implicit": torch.full((2, 6), 1.5),
    "perturb_body_materials": torch.full((2, 3), 0.7),
    "random_joint_offset": torch.full((2, 2), 0.01),
    "perturb_gravity": torch.tensor([[0.0, 0.0, -9.7], [0.1, 0.0, -9.8]]),
  }
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    event_manager=_FakeEventManager(
      {name: _ObservedTerm(value) for name, value in values.items()}
    ),
    scene={"robot": SimpleNamespace(joint_names=("j0", "j1"))},
  )

  assert torch.equal(sp_mdp.domain_motor_params_implicit(env), values["motor_params_implicit"])
  assert torch.equal(sp_mdp.domain_perturb_body_materials(env), values["perturb_body_materials"])
  assert torch.equal(sp_mdp.domain_random_joint_offset(env), values["random_joint_offset"])
  assert torch.equal(sp_mdp.domain_perturb_gravity(env), values["perturb_gravity"])


class _FakeRobot:
  body_names = sp_mdp.SP_REQUIRED_BODY_NAMES

  def __init__(self, current_z: torch.Tensor):
    body_pos = torch.zeros((current_z.shape[0], len(self.body_names), 3))
    body_pos[..., 2] = current_z
    self.data = SimpleNamespace(body_link_pos_w=body_pos)

  def find_bodies(self, names, preserve_order: bool = True):
    del preserve_order
    body_ids = []
    for name in names:
      if name in self.body_names:
        body_ids.append(self.body_names.index(name))
        continue
      token = name.replace(".*", "")
      body_ids.extend(
        index for index, body_name in enumerate(self.body_names) if token in body_name
      )
    return body_ids, [self.body_names[index] for index in body_ids]


class _FakeMotionCommand:
  def __init__(self, target_z: torch.Tensor):
    self.cfg = SimpleNamespace(body_names=sp_mdp.SP_REQUIRED_BODY_NAMES)
    self.motion_idx = torch.zeros(target_z.shape[0], dtype=torch.long)
    self.time_steps = torch.zeros(target_z.shape[0], dtype=torch.long)
    body_pos = torch.zeros((target_z.shape[0], len(self.cfg.body_names), 3))
    body_pos[..., 2] = target_z
    self._body_pos = body_pos

  def _gather_motion_field(
    self, field_name: str, motion_idx: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    assert field_name == "body_pos_w"
    assert motion_idx.shape == (self._body_pos.shape[0],)
    assert time_steps.shape == (self._body_pos.shape[0], 1)
    return self._body_pos.unsqueeze(1)


class _FakeCommandManager:
  def __init__(self, command):
    self.command = command

  def get_term(self, name: str):
    assert name == "motion"
    return self.command


def _termination_env(*, current_z: float, target_z: float):
  target = torch.full((1, len(sp_mdp.SP_REQUIRED_BODY_NAMES)), target_z)
  current = torch.full((1, len(sp_mdp.SP_REQUIRED_BODY_NAMES)), current_z)
  command = _FakeMotionCommand(target)
  return SimpleNamespace(
    num_envs=1,
    device="cpu",
    command_manager=_FakeCommandManager(command),
    scene={"robot": _FakeRobot(current)},
  )


def test_body_z_termination_uses_target_relative_height() -> None:
  env = _termination_env(current_z=1.0, target_z=1.0)

  done = sp_mdp.body_z_termination(
    env,
    command_name="motion",
    body_z_terminate_thres=(-0.25, 0.35),
    body_z_terminate_patterns=("pelvis", "head_mimic", ".*_hand_mimic", ".*ankle_roll_link"),
  )

  assert torch.equal(done, torch.tensor([False]))


def test_body_z_termination_requires_continuous_exceed_frames() -> None:
  env = _termination_env(current_z=1.5, target_z=1.0)

  outputs = [
    sp_mdp.body_z_termination(
      env,
      command_name="motion",
      body_z_terminate_thres=(-0.25, 0.35),
      body_z_terminate_patterns=("pelvis", "head_mimic", ".*_hand_mimic", ".*ankle_roll_link"),
    )
    for _ in range(5)
  ]

  assert [bool(output.item()) for output in outputs] == [False, False, False, False, True]


def test_failure_terminations_respect_source_reset_warmup() -> None:
  env = SimpleNamespace(episode_length_buf=torch.tensor([5, 6]))
  command = SimpleNamespace(
    cfg=SimpleNamespace(termination_warmup_steps=5)
  )

  gated = sp_mdp._apply_termination_warmup(
    env, command, torch.tensor([True, True])
  )

  assert torch.equal(gated, torch.tensor([False, True]))


def test_loco_reward_group_schedule_matches_source_pretrain_factor() -> None:
  names = ("survival", "action_rate_l2")
  base = {"survival": 3.0, "action_rate_l2": 0.02}
  term_cfgs = {name: SimpleNamespace(weight=weight) for name, weight in base.items()}
  manager = SimpleNamespace(get_term_cfg=lambda name: term_cfgs[name])
  env = SimpleNamespace(
    scene={"robot": SimpleNamespace()},
    reward_manager=manager,
    num_envs=2,
    device="cpu",
  )
  cfg = SimpleNamespace(
    params={
      "term_names": names,
      "base_weights": base,
      "progress_range": (0.0, 1.0),
      "factor_range": (0.5, 1.0),
    }
  )
  schedule = sp_mdp.loco_reward_group_schedule(cfg, env)

  assert schedule.step_schedule(0.0)["factor"] == 0.5
  assert term_cfgs["survival"].weight == 1.5
  assert term_cfgs["action_rate_l2"].weight == 0.01
  assert schedule.step_schedule(1.0)["factor"] == 1.0
  assert term_cfgs["survival"].weight == 3.0
  assert term_cfgs["action_rate_l2"].weight == 0.02
