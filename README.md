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
- `src/sp_tracking/assets/robots`: packaged G1 assets selected by task. The
  ablations use the SP XML/body topology with BFM init, collision, and actuator
  settings; `tracking_bfm_sp` retains the complete HEFT robot configuration.
- `scripts/train_tracking_bfm.sh`: repo-local launch script for the default
  training run.
- `scripts/play_tracking_bfm.sh`: repo-local play script for local or W&B
  checkpoints.

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

The SP profile is a teacher-only HEFT-style pretrain: the actor consumes
`policy + priv`, encodes `priv` to 256 dimensions, and trains with the matching
teacher network, optimizer, schedules, symmetry and motion randomization. Start
pretraining from scratch with:

```bash
uv run sp-train task=tracking_bfm_sp motion_path=/path/to/motions
```

The resulting checkpoint is self-describing and can be passed directly to
`sp-play`. This profile intentionally does not implement the student/adapt
stage. The original `tracking_bfm` dynamics and MLP/PPO preset are retained;
all task observation and reward views now use a pelvis anchor.

The three observation ablations use the SP XML only for body/joint/reference
compatibility. They otherwise retain the complete BFM runtime: BFM MLPs,
PPO/Adam, action, rewards, terminations, events, curriculum, sampler, reset,
seed, and training budget. Run them with:

```bash
uv run sp-train task=tracking_bfm_sp_ablation_bfm_actor motion_path=/path/to/motions
uv run sp-train task=tracking_bfm_sp_ablation_student_actor motion_path=/path/to/motions
uv run sp-train task=tracking_bfm_sp_ablation_teacher_actor motion_path=/path/to/motions
```

The default `tracking_bfm` task keeps the BFM tracking term set, termination,
and `multi_commands.py` loader behavior, with its observation/reward anchor
standardized to pelvis. `tracking_bfm_sp` remains the independent HEFT pretrain
profile.

## Tasks

| Task | Actor observation | Critic observation | Runtime |
| --- | --- | --- | --- |
| `tracking_bfm` | BFM `actor` | BFM `critic` | BFM XML + complete BFM runtime |
| `tracking_bfm_sp` | `policy + priv` | `policy + priv + priv_critic` | SP XML + HEFT pretrain runtime |
| `tracking_bfm_sp_ablation_bfm_actor` | BFM `actor` | `policy + priv` | SP XML compatibility + complete BFM runtime |
| `tracking_bfm_sp_ablation_student_actor` | SP student `policy` | `policy + priv` | SP XML compatibility + complete BFM runtime |
| `tracking_bfm_sp_ablation_teacher_actor` | raw `policy + priv` | `policy + priv` | SP XML compatibility + complete BFM runtime |

All three ablations use a raw-observation BFM `MLPModel` actor with no adapter
or privileged encoder. Only the input-facing hidden width is adjusted so the
three actor parameter counts remain within 0.03% of the 8.62M-parameter BFM
baseline. Their critic is the same `HeftTeacherCritic` with hidden dimensions
`[1024, 512, 512]`, Mish activation, and `vecnorm_decay=0.9999`.
The ablations omit `priv_critic` because its four terms expose HEFT-specific
domain-randomization state; under BFM randomization they would only be constant
fallback values. The independent `tracking_bfm_sp` pretrain task keeps them.

## Checkpoints, Resume, and Deployment Export

Checkpoints follow the reference-compatible naming convention:

```text
checkpoint_<iteration>.pt
checkpoint_final.pt
policy.onnx
policy.json
deploy_metadata.json
cfg.yaml
config.yaml
```

`checkpoint_*.pt` contains the RSL-RL training state for resume plus the
reference-compatible checkpoint fields `policy`, `env`, `cfg`, and `vecnorm`.
The actor state remains the authoritative normalizer source for RSL-RL; the
`vecnorm` field is a compatible source-style view for checkpoint tooling.
Training resume is local-only:

```bash
uv run sp-train checkpoint_path=/path/to/checkpoint_final.pt
uv run sp-train agent.resume=true agent.load_run='.*' agent.load_checkpoint='checkpoint_.*.pt'
```

`policy.onnx` is exported after every save and is intentionally deployment
oriented: input key `policy`, output key `action`, with `policy.json` sidecar
metadata matching the reference ONNX runtime loader.

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

## Play

Play is local-checkpoint-only. New checkpoints embed the resolved training
configuration, so `--task` is inferred and observation/reward/network
ablations are reconstructed from the checkpoint. Pass `--task` only for a
legacy checkpoint that lacks `cfg`.

Use the play script with a local checkpoint:

```bash
scripts/play_tracking_bfm.sh \
  --checkpoint-file logs/rsl_rl/g1_tracking/<run>/checkpoint_final.pt \
  --motion-file /path/to/motion.npz
```

For large-dataset play/debug:

```bash
scripts/play_tracking_bfm.sh \
  --checkpoint-file logs/rsl_rl/g1_tracking/<run>/checkpoint_final.pt \
  --motion-path /path/to/motions
```

The script forwards to `uv run sp-play` and supports `--num-envs`, `--viewer`,
and `--domain-randomization`. W&B remains logging-only; model checkpoints are
not uploaded or downloaded through it.

## Torque Safety

All packaged G1 tasks use the same conservative sim2real effort limits:
hip yaw / waist yaw 88, hip pitch 88, hip roll / knee 139,
ankle pitch / roll 35, waist pitch / roll 35, shoulder / elbow / wrist roll 25,
and wrist pitch / yaw 5. The motion-tracking action curriculum also clamps
scheduled torque scaling so it cannot exceed the safe asset limits.

## Development Checks

```bash
uv run python -m pytest -q
uv build
```

## mjlab Task Entry Point

Installing the package exposes eight mjlab task IDs via the `mjlab.tasks` entry
point:

- `SPTracking-G1-BFM-BFMActor-BFMCritic`
- `SPTracking-G1-HEFT-TeacherActor-HEFTCritic`
- `SPTracking-G1-BFM-BFMActor-HEFTCritic`
- `SPTracking-G1-BFM-StudentActor-HEFTCritic`
- `SPTracking-G1-BFM-TeacherActor-HEFTCritic`
- `SPTracking-G1-BFM-StudentActor-BFMCritic`
- `SPTracking-G1-BFM-TeacherActor-BFMCritic`
- `SPTracking-G1-BFM-WBTeleopActor-BFMCritic`
