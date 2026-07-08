# SP Tracking

External Hydra-configured tracking tasks for `mjlab==1.5.0`.

The package keeps tracking-specific command, observation, reward, termination,
and rsl-rl training code outside the mjlab source tree.

## Layout

- `src/sp_tracking/tasks/tracking/mdp`: tracking command loaders, observations,
  rewards, terminations, and metrics.
- `src/sp_tracking/config`: Python builders that translate Hydra YAML into
  mjlab manager/rsl-rl dataclass configs.
- `src/sp_tracking/conf`: packaged Hydra configs used by `sp-train`.
- `conf`: editable mirror of the packaged configs. Tests assert both copies are
  identical.

## Training

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

## mjlab Task Entry Point

Installing the package exposes two mjlab task IDs via the `mjlab.tasks` entry
point:

- `SPTracking-G1-BFM`
- `SPTracking-G1-BFM-SP`
