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


def get_wandb_checkpoint_path(
  *, log_root: str | Path, run_path: str | Path, checkpoint_name: str | None = None
) -> tuple[Path, bool]:
  """Download a model checkpoint from a W&B run, caching it under log_root."""
  import wandb

  run_path = Path(run_path)
  run_id = str(run_path).split("/")[-1]
  download_dir = Path(log_root) / "wandb_checkpoints" / run_id

  api = wandb.Api()
  wandb_run = api.run(str(run_path))
  files = [
    file.name
    for file in wandb_run.files()
    if file.name.startswith("model_") and file.name.endswith(".pt")
  ]
  if checkpoint_name is None:
    if not files:
      raise ValueError(f"No model checkpoints found in W&B run {run_path}")
    checkpoint_file = sorted(files, key=checkpoint_sort_key)[-1]
  else:
    if checkpoint_name not in files:
      raise ValueError(
        f"Checkpoint '{checkpoint_name}' not found in run {run_path}. Available: {files}"
      )
    checkpoint_file = checkpoint_name

  checkpoint_path = download_dir / checkpoint_file
  was_cached = checkpoint_path.exists()
  if not was_cached:
    download_dir.mkdir(parents=True, exist_ok=True)
    wandb_run.file(checkpoint_file).download(str(download_dir), replace=True)
  return checkpoint_path, was_cached
