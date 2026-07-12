from __future__ import annotations

import re
from pathlib import Path


_ITER_PATTERN = re.compile(r"^(?:model|checkpoint)_(\d+)\.pt$")
_FINAL_NAMES = {"model_final.pt", "checkpoint_final.pt"}


def checkpoint_iteration(path: str | Path) -> int | None:
  """Return the numeric iteration encoded in a checkpoint filename."""
  name = Path(path).name
  if name in _FINAL_NAMES:
    return None
  match = _ITER_PATTERN.match(name)
  if match is None:
    return None
  return int(match.group(1))


def checkpoint_sort_key(path: str | Path) -> tuple[int, int, str]:
  """Sort checkpoints by training iteration, keeping final checkpoints last."""
  name = Path(path).name
  if name in _FINAL_NAMES:
    return (1, 0, name)
  iteration = checkpoint_iteration(name)
  if iteration is None:
    return (0, -1, name)
  return (0, iteration, name)


def resolve_local_checkpoint_path(
  *, log_root: str | Path, load_run: str, load_checkpoint: str
) -> Path:
  """Resolve a checkpoint from a local experiment log root."""
  root = Path(log_root)
  if not root.exists():
    raise ValueError(f"Log path does not exist: {root}")

  run_re = re.compile(load_run)
  checkpoint_re = re.compile(load_checkpoint)
  runs = [
    path
    for path in root.iterdir()
    if path.is_dir() and path.name != "wandb_checkpoints" and run_re.match(path.name)
  ]
  if not runs:
    raise ValueError(f"No run directories found in {root} matching '{load_run}'")
  run_path = sorted(runs)[-1]

  checkpoints = [
    path for path in run_path.iterdir() if path.is_file() and checkpoint_re.match(path.name)
  ]
  if not checkpoints:
    raise ValueError(f"No checkpoint found in {run_path} matching {load_checkpoint}")
  return sorted(checkpoints, key=checkpoint_sort_key)[-1]
