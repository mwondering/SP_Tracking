"""HEFT-compatible Muon/AdamW optimizer construction."""

from __future__ import annotations

import warnings

import torch


def _unique(parameters):
  seen: set[int] = set()
  result = []
  for parameter in parameters:
    if parameter.requires_grad and id(parameter) not in seen:
      seen.add(id(parameter))
      result.append(parameter)
  return result


class OptimizerGroup(torch.optim.Optimizer):
  def __init__(self, optimizers: list[torch.optim.Optimizer]):
    parameters = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
    super().__init__(parameters, defaults={})
    self.optimizers = optimizers
    self.param_groups = [group for opt in optimizers for group in opt.param_groups]

  def zero_grad(self, set_to_none: bool | None = None):
    for optimizer in self.optimizers:
      optimizer.zero_grad() if set_to_none is None else optimizer.zero_grad(set_to_none=set_to_none)

  @torch.no_grad()
  def step(self, closure=None):
    result = None
    for index, optimizer in enumerate(self.optimizers):
      current = optimizer.step(closure if index == 0 else None)
      result = current if result is None else result
    return result

  def state_dict(self):
    return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

  def load_state_dict(self, state_dict):
    states = state_dict.get("optimizers", ())
    if len(states) != len(self.optimizers):
      warnings.warn("optimizer group size differs while loading checkpoint")
    for optimizer, state in zip(self.optimizers, states):
      optimizer.load_state_dict(state)


def build_heft_optimizer(parameters, *, lr: float, adamw_only=()):
  parameters = _unique(parameters)
  adamw_ids = {id(parameter) for parameter in _unique(adamw_only)}
  muon = [p for p in parameters if p.ndim == 2 and id(p) not in adamw_ids]
  adamw = [p for p in parameters if p.ndim != 2 or id(p) in adamw_ids]
  optimizers: list[torch.optim.Optimizer] = []
  if adamw:
    optimizers.append(torch.optim.AdamW(adamw, lr=lr, weight_decay=0.0))
  if muon:
    optimizers.append(
      torch.optim.Muon(muon, lr=lr, adjust_lr_fn="match_rms_adamw", weight_decay=0.0)
    )
  if len(optimizers) == 1:
    return optimizers[0]
  return OptimizerGroup(optimizers)
