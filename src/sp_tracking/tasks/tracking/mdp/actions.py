from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if False:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MotionTrackingJointPositionActionCfg(JointPositionActionCfg):
  max_delay: int = 2
  delay_full_progress: float = 0.8
  alpha: tuple[float, float] = (0.8, 1.0)
  torque_limit_scale_range: tuple[float, float] = (4.0, 1.0)
  torque_limit_progress_range: tuple[float, float] = (0.0, 0.8)

  def build(self, env: "ManagerBasedRlEnv") -> "MotionTrackingJointPositionAction":
    return MotionTrackingJointPositionAction(self, env)


class MotionTrackingJointPositionAction(JointPositionAction):
  cfg: MotionTrackingJointPositionActionCfg

  def __init__(self, cfg: MotionTrackingJointPositionActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    if not isinstance(self._offset, torch.Tensor):
      self._offset = torch.full_like(self._processed_actions, float(self._offset))
    self._default_offset = self._offset.clone()
    self.joint_offset = torch.zeros_like(self._processed_actions)
    self.applied_action = torch.zeros_like(self._raw_actions)
    self.alpha = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)

    max_delay_steps = max(int(cfg.max_delay), 0)
    self.max_delay = max_delay_steps
    self._decimation = int(env.cfg.decimation)
    self._history_len = max(
      8, (max_delay_steps + self._decimation - 1) // self._decimation + 1
    )
    self._action_history = torch.zeros(
      (self.num_envs, self._history_len, self.action_dim),
      dtype=torch.float32,
      device=self.device,
    )
    self.delay = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device)
    self.delay_probs = torch.zeros(max_delay_steps + 1, dtype=torch.float32, device=self.device)
    self._substep = 0

    if "actuator_forcerange" not in env.sim.expanded_fields:
      env.sim.expand_model_fields(("actuator_forcerange",))
    self._ctrl_ids = self._entity.indexing.ctrl_ids[self._target_ids]
    self._default_forcerange = env.sim.get_default_field("actuator_forcerange")[
      self._ctrl_ids
    ].clone()
    self._torque_limit_scale: float | None = None
    self.step_schedule(0.0, None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    self._action_history[env_ids] = 0.0
    self.applied_action[env_ids] = 0.0
    self._substep = 0
    if isinstance(env_ids, slice):
      count = self.num_envs
    else:
      count = len(env_ids)
    sampled_delay = torch.multinomial(self.delay_probs, count, replacement=True)
    self.delay[env_ids] = sampled_delay.unsqueeze(-1).to(self.delay.dtype)
    alpha_low, alpha_high = self.cfg.alpha
    self.alpha[env_ids] = torch.empty((count, 1), device=self.device).uniform_(
      float(alpha_low), float(alpha_high)
    )

  def set_joint_offset(self, env_ids: torch.Tensor, offset: torch.Tensor) -> None:
    self.joint_offset[env_ids] = offset

  def get_recent_actions(self, steps: int) -> torch.Tensor:
    if steps <= self._history_len:
      return self._action_history[:, :steps]
    pad = torch.zeros(
      (self.num_envs, steps - self._history_len, self.action_dim),
      dtype=self._action_history.dtype,
      device=self.device,
    )
    return torch.cat((self._action_history, pad), dim=1)

  def get_recent_action_obs(self, steps: int) -> torch.Tensor:
    return self.get_recent_actions(steps)

  def get_recent_action_rate_actions(self, steps: int) -> torch.Tensor:
    return self.get_recent_actions(steps)

  def step_schedule(self, progress: float, iters: int | None = None) -> dict[str, float]:
    del iters
    self._schedule_delay(progress)
    scale = self._schedule_torque_limit(progress)
    return {
      "progress": float(progress),
      "torque_limit_scale": float(scale),
      "max_delay_probability": float(self.delay_probs[-1].item()),
    }

  def _schedule_delay(self, progress: float) -> None:
    max_delay = int(self.max_delay)
    if max_delay <= 0:
      self.delay_probs.fill_(1.0)
      return
    full_progress = max(float(self.cfg.delay_full_progress), 1.0e-6)
    p = max(0.0, min(float(progress) / full_progress, 1.0))
    q = max(0.0, min(float(max_delay), p * float(max_delay + 1) - 1.0))
    k = int(q)
    alpha = q - float(k)
    kp1 = min(k + 1, max_delay)
    self.delay_probs.zero_()
    self.delay_probs[: k + 1] += (1.0 - alpha) / float(k + 1)
    if kp1 > k:
      self.delay_probs[: kp1 + 1] += alpha / float(kp1 + 1)
    self.delay_probs /= self.delay_probs.sum().clamp_min(1.0e-8)

  def _schedule_torque_limit(self, progress: float) -> float:
    start, end = self.cfg.torque_limit_progress_range
    start_scale, end_scale = self.cfg.torque_limit_scale_range
    p = max(0.0, min(float(progress), 1.0))
    if p <= float(start):
      scale = float(start_scale)
    elif p >= float(end) or abs(float(end) - float(start)) <= 1.0e-9:
      scale = float(end_scale)
    else:
      ratio = (p - float(start)) / (float(end) - float(start))
      scale = float(start_scale) + ratio * (float(end_scale) - float(start_scale))
    if self._torque_limit_scale is not None and abs(scale - self._torque_limit_scale) < 1.0e-6:
      return scale
    force_range = self._default_forcerange.unsqueeze(0) * scale
    self._env.sim.model.actuator_forcerange[:, self._ctrl_ids] = force_range
    self._torque_limit_scale = scale
    return scale

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._action_history = torch.roll(self._action_history, shifts=1, dims=1)
    self._action_history[:, 0] = self._raw_actions
    self._substep = 0

  def _update_processed_actions(self, substep: int) -> None:
    delay_env_steps = torch.div(
      self.delay.squeeze(-1) - int(substep) + self._decimation - 1,
      self._decimation,
      rounding_mode="floor",
    ).clamp(0, self._history_len - 1)
    env_ids = torch.arange(self.num_envs, device=self.device)
    delayed_action = self._action_history[env_ids, delay_env_steps]
    self.applied_action.lerp_(delayed_action, self.alpha)
    self._processed_actions = (
      self.applied_action * self._scale + self._default_offset + self.joint_offset
    )
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )

  def apply_actions(self) -> None:
    self._update_processed_actions(self._substep)
    super().apply_actions()
    self._substep = (self._substep + 1) % self._decimation
