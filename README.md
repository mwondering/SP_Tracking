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
  BFM-runtime comparisons use the BFM XML and express HEFT keypoints through
  physical BFM body-frame transforms; `tracking_bfm_sp` retains the complete
  HEFT robot configuration.
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

The three observation ablations use the BFM XML and add only the reference
caches, semantic keypoint views, contact sensing, and policy-mean history needed
to construct HEFT observations. They retain the BFM MLP actor family, PPO/Adam,
applied joint-position dynamics, rewards, terminations, events, curriculum,
sampler, reset, seed, and training budget. Run them with:

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

| Task | Actor observation | Critic observation | Reward | Runtime |
| --- | --- | --- | --- | --- |
| `tracking_bfm` | BFM `actor` | BFM `critic` | BFM | BFM XML + complete BFM runtime |
| `tracking_bfm_sp` | `policy + priv` | `policy + priv + priv_critic` | HEFT | SP XML + HEFT pretrain runtime |
| `tracking_bfm_sp_ablation_bfm_actor` | BFM `actor` | `policy + priv` | BFM | BFM XML/runtime + HEFT observation support |
| `tracking_bfm_sp_ablation_student_actor` | HEFT student `policy` | `policy + priv` | BFM | BFM XML/runtime + HEFT observation support |
| `tracking_bfm_sp_ablation_teacher_actor` | raw `policy + priv` | `policy + priv` | BFM | BFM XML/runtime + HEFT observation support |
| `tracking_bfm_student_actor_bfm_critic` | HEFT student `policy` | BFM `critic` | BFM | BFM XML/runtime + HEFT observation support |
| `tracking_bfm_teacher_actor_bfm_critic` | raw `policy + priv` | BFM `critic` | BFM | BFM XML/runtime + HEFT observation support |
| `tracking_bfm_wbteleop_actor_bfm_critic` | deployable WBTeleop `actor` (886-D) | BFM `critic` | BFM | BFM XML + complete BFM runtime |
| `tracking_bfm_wbteleop_actor_heft_critic` | deployable WBTeleop `actor` (886-D) | `policy + priv` | BFM | BFM XML/runtime + HEFT observation support |
| `tracking_bfm_spv1_actor_heft_critic_heft_reward` | heading-invariant SPV1 `actor` (1786-D) | `policy + priv` | HEFT | BFM XML/runtime + measured joint-torque sensors |
| `tracking_bfm_spv2_actor_heft_critic_heft_reward` | Compact SPV2: 5-frame history, +4 future, HEFT root rotation (1056-D) | `policy + priv` | HEFT | BFM XML/runtime + measured joint-torque sensors |

Every BFM-XML task above except the already-HEFT-reward SPV tasks also has an
additive HEFT-reward variant. Append `_heft_reward` to its Hydra task name, for
example:

```bash
uv run sp-train task=tracking_bfm_wbteleop_actor_heft_critic_heft_reward \
  motion_path=/path/to/motions
```

This defines eight additional tasks:

- `tracking_bfm_heft_reward`
- `tracking_bfm_sp_ablation_bfm_actor_heft_reward`
- `tracking_bfm_sp_ablation_student_actor_heft_reward`
- `tracking_bfm_sp_ablation_teacher_actor_heft_reward`
- `tracking_bfm_student_actor_bfm_critic_heft_reward`
- `tracking_bfm_teacher_actor_bfm_critic_heft_reward`
- `tracking_bfm_wbteleop_actor_bfm_critic_heft_reward`
- `tracking_bfm_wbteleop_actor_heft_critic_heft_reward`

Each variant inherits its parent actor, critic, action, PPO, events,
terminations, curriculum, simulation, and seed. Only the reward is replaced,
plus the internal BFM reference bodies, foot contact sensor, and substep cache
needed to evaluate it. The original nine task configurations are unchanged.
The complete SP task is not duplicated because `tracking_bfm_sp` already uses
the HEFT reward and HEFT runtime.

The HEFT preset contains the ten tracking terms and seven enabled locomotion
terms from the source pretrain config. Its locomotion group starts at factor
0.5 and is scheduled linearly to 1.0; MJLab applies the same per-step `dt`
scaling as the source. On the BFM XML, missing SP-only bodies are represented
by kinematically exact semantic points: the hand includes the 5 mm wrist-chain
correction and the toe is the ankle-roll frame plus 0.1 m along local x.

All three ablations use a raw-observation BFM `MLPModel` actor with no adapter
or privileged encoder. The BFM and teacher actors retain parameter-matched
input widths; the student actor deliberately uses a 2048-wide first layer so
its wider observation is not prematurely compressed. Its policy-facing action
vector is also placed in the same HEFT canonical joint order as its target,
joint-history, and previous-action observations, while physical BFM joint
targets and dynamics remain unchanged. Their critic is the same
`HeftTeacherCritic` with hidden dimensions
`[1024, 512, 512]`, Mish activation, and `vecnorm_decay=0.9999`.
The ablations omit `priv_critic` because its four terms expose HEFT-specific
domain-randomization state; under BFM randomization they would only be constant
fallback values. The independent `tracking_bfm_sp` pretrain task keeps them.

The two WBTeleop tasks share the exact same actor term order, history lengths,
BFM PPO, reward, termination, and joint-position action mapping. Their actor is
58-D current reference joint state, 180-D reference limb pose history, 3-D
reference angular velocity, 180-D measured limb pose history, and five-frame
deployable proprioception (gravity, gyro, joint position/velocity, last action),
for 886 dimensions total. It contains no `base_lin_vel` or global tracking-error
term. The HEFT-critic variant additionally builds `policy + priv`; the extra
cache, contact sensor, and action-mean history are observation support and do
not enter the actor or BFM reward.

The SPV1 actor combines ten-frame deployable proprioception (joint state,
gravity, gyro, raw action, and four-substep-averaged `jointactuatorfrc`) with
six-step heading-invariant root offsets, current-through-six-step joint,
gravity, and angular-velocity commands, and explicit current
reference-minus-measurement errors.
Neither global root position nor robot global yaw enters the actor. The task
uses the unchanged HEFT `policy + priv` critic and the complete HEFT reward on
the BFM runtime.

SPV2 compresses all proprioceptive histories to five consecutive frames and all
reference lookahead to the current frame plus four future frames. It retains
reference pelvis height and reference-local pelvis linear velocity, and changes
the root-orientation command to HEFT's noisy robot-to-reference 6D rotation.
The resulting actor is 1056-D; it contains relative yaw but no robot global
position.

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

Installing the package exposes nine mjlab task IDs via the `mjlab.tasks` entry
point:

- `SPTracking-G1-BFM-BFMActor-BFMCritic`
- `SPTracking-G1-HEFT-TeacherActor-HEFTCritic`
- `SPTracking-G1-BFM-BFMActor-HEFTCritic`
- `SPTracking-G1-BFM-StudentActor-HEFTCritic`
- `SPTracking-G1-BFM-TeacherActor-HEFTCritic`
- `SPTracking-G1-BFM-StudentActor-BFMCritic`
- `SPTracking-G1-BFM-TeacherActor-BFMCritic`
- `SPTracking-G1-BFM-WBTeleopActor-BFMCritic`
- `SPTracking-G1-BFM-WBTeleopActor-HEFTCritic`
