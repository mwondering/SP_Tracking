# SP Tracking

External Hydra-configured tracking tasks for `mjlab==1.5.0`.

The package keeps tracking-specific command, observation, reward, termination,
and rsl-rl training code outside the mjlab source tree.

## Quick Start

```bash
uv sync
scripts/train_tracking_bfm.sh
```

The default training script runs `tracking_bfm` with the multi-motion loader and
uses `/home/lenovo/DATASETS/Data10k_full` as the motion path. Override it either
with the first positional argument or with `SP_TRACKING_MOTION_PATH`:

```bash
scripts/train_tracking_bfm.sh /path/to/motions
SP_TRACKING_MOTION_PATH=/path/to/motions scripts/train_tracking_bfm.sh
```

Hydra overrides can be appended after the motion path:

```bash
scripts/train_tracking_bfm.sh /path/to/motions task.num_envs=2048 agent.max_iterations=50000
```

## Layout

- `src/sp_tracking/tasks/tracking/mdp`: tracking command loaders, observations,
  rewards, terminations, and metrics.
- `src/sp_tracking/config`: Python builders that translate Hydra YAML into
  mjlab manager/rsl-rl dataclass configs.
- `src/sp_tracking/conf`: the single Hydra config source used by `sp-train`,
  tests, and the mjlab task entry point.
- `src/sp_tracking/assets/robots`: two packaged G1 assets selected by task:
  `tracking_bfm_g1` for `tracking_bfm` and `motion_tracking_g1` for
  `tracking_bfm_sp`.
- `scripts/train_tracking_bfm.sh`: repo-local launch script for the default
  training run.

## Training

Use the launch script for normal training:

```bash
scripts/train_tracking_bfm.sh
```

Useful environment overrides:

```bash
SP_TRACKING_NUM_ENVS=4096 \
SP_TRACKING_RUN_NAME=bfm_4096env \
SP_TRACKING_WANDB_PROJECT=sp-tracking \
scripts/train_tracking_bfm.sh /path/to/motions
```

Use the torchrun launch script for multi-GPU training:

```bash
SP_TRACKING_GPUS=0,1 \
SP_TRACKING_NUM_ENVS=4096 \
scripts/train_tracking_bfm_multigpu.sh /path/to/motions
```

`SP_TRACKING_NUM_ENVS` and `task.num_envs` are per-rank values. With two GPUs
and `SP_TRACKING_NUM_ENVS=4096`, each GPU creates 4096 environments and the
global rollout batch is doubled. Override `SP_TRACKING_NPROC` only when it
should differ from the number of comma-separated GPU IDs in `SP_TRACKING_GPUS`.

The script forwards to:

```bash
uv run sp-train task=tracking_bfm motion_path=/path/to/motions
```

You can also call the Hydra entry point directly:

```bash
uv sync
uv run sp-train motion_path=/path/to/motions task.num_envs=4096
```

Use the SP large-dataset/observation/reward variant:

```bash
uv run sp-train task=tracking_bfm_sp motion_path=/path/to/motions
```

The default `tracking_bfm` task keeps the old BFM tracking observation, reward,
termination, and `multi_commands.py` loader defaults. `tracking_bfm_sp` switches
to `multi_command_largedataset.py` plus the SP observation/reward/termination
sets for ablations and debugging.

## Tasks

| Task | Loader | Observation/Reward Set | Robot Asset |
| --- | --- | --- | --- |
| `tracking_bfm` | `multi_commands.py` | old BFM tracking defaults | `tracking_bfm_g1` |
| `tracking_bfm_sp` | `multi_command_largedataset.py` | SP tracking defaults | `motion_tracking_g1` |

Both tasks are configured under `src/sp_tracking/conf/task`. Shared PPO defaults
live in `src/sp_tracking/conf/agent/tracking_bfm_ppo.yaml`.

## Logging

Training logs are written under `logs/rsl_rl` by default. The launch script also
passes `launch_script_path` to `sp-train`; the training entry copies the exact
script used for the run into:

```text
logs/rsl_rl/g1_tracking/<run>/launch/train_tracking_bfm.sh
```

When wandb logging is enabled, the copied launch script is uploaded through the
rsl-rl log writer as a run file. The default script sets `agent.logger=wandb`;
model checkpoint upload can be controlled with the `agent.upload_model` Hydra
override.

## Development Checks

```bash
uv run python -m pytest -q
uv build
```

## mjlab Task Entry Point

Installing the package exposes two mjlab task IDs via the `mjlab.tasks` entry
point:

- `SPTracking-G1-BFM`
- `SPTracking-G1-BFM-SP`
