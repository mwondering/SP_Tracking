# SP Tracking

External Hydra-configured tracking tasks for `mjlab==1.5.0`.

The package keeps tracking-specific command, observation, reward, termination,
and rsl-rl training code outside the mjlab source tree.

## Quick Start

```bash
uv sync
scripts/train_tracking_bfm.sh
```

The default training script runs `SPTracking-G1-BFM-BFMActor-BFMCritic` with the multi-motion loader and
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

The script forwards the canonical public task ID to:

```bash
uv run sp-train task_id=SPTracking-G1-BFM-BFMActor-BFMCritic motion_path=/path/to/motions
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
uv run sp-train task_id=SPTracking-G1-HEFT-TeacherActor-HEFTCritic motion_path=/path/to/motions
```

The resulting checkpoint is self-describing and can be passed directly to
`sp-play`. This profile intentionally does not implement the student/adapt
stage. The original `tracking_bfm` dynamics and MLP/PPO preset are retained;
all task observation and reward views now use a pelvis anchor.

An official-code-compatible SAPG extension can be enabled on any task with
`agent.algorithm.sapg_cfg.enabled=true`. It is disabled by default and removed
before the original PPO constructor is called, so existing PPO runs retain
their previous model, optimizer and update paths. Configuration, algorithm
semantics, checkpoint behavior and current limitations are documented in
[docs/sapg.md](docs/sapg.md).

The two-motion SPV5 gradient-conflict experiment is available as
`SPTracking-G1-TestPolicyGradients`. Its three-run protocol, exact gradient
definitions, SQLite schema, and analysis guidance are documented in
[docs/policy_gradient_diagnostics.md](docs/policy_gradient_diagnostics.md).

The three observation ablations use the BFM XML and add only the reference
caches, semantic keypoint views, contact sensing, and policy-mean history needed
to construct HEFT observations. They retain the BFM MLP actor family, PPO/Adam,
applied joint-position dynamics, rewards, terminations, events, curriculum,
sampler, reset, seed, and training budget. Run them with:

```bash
uv run sp-train task_id=SPTracking-G1-BFM-BFMActor-HEFTCritic motion_path=/path/to/motions
uv run sp-train task_id=SPTracking-G1-BFM-StudentActor-HEFTCritic motion_path=/path/to/motions
uv run sp-train task_id=SPTracking-G1-BFM-TeacherActor-HEFTCritic motion_path=/path/to/motions
```

The default `tracking_bfm` task keeps the BFM tracking term set, termination,
and `multi_commands.py` loader behavior, with its observation/reward anchor
standardized to pelvis. `tracking_bfm_sp` remains the independent HEFT pretrain
profile.

## Tasks

| Task | Actor observation | Critic observation | Reward | Runtime |
| --- | --- | --- | --- | --- |
| `SPTracking-G1-BFM-BFMActor-BFMCritic` | BFM `actor` | BFM `critic` | BFM | BFM XML + complete BFM runtime |
| `SPTracking-G1-HEFT-TeacherActor-HEFTCritic` | `policy + priv` | `policy + priv + priv_critic` | HEFT | SP XML + HEFT pretrain runtime |
| `SPTracking-G1-BFM-BFMActor-HEFTCritic` | BFM `actor` | `policy + priv` | BFM | BFM XML/runtime + HEFT observation support |
| `SPTracking-G1-BFM-StudentActor-HEFTCritic` | HEFT student `policy` | `policy + priv` | BFM | BFM XML/runtime + HEFT observation support |
| `SPTracking-G1-BFM-TeacherActor-HEFTCritic` | raw `policy + priv` | `policy + priv` | BFM | BFM XML/runtime + HEFT observation support |
| `SPTracking-G1-BFM-StudentActor-BFMCritic` | HEFT student `policy` | BFM `critic` | BFM | BFM XML/runtime + HEFT observation support |
| `SPTracking-G1-BFM-TeacherActor-BFMCritic` | raw `policy + priv` | BFM `critic` | BFM | BFM XML/runtime + HEFT observation support |
| `SPTracking-G1-BFM-WBTeleopActor-BFMCritic` | deployable WBTeleop `actor` (886-D) | BFM `critic` | BFM | BFM XML + complete BFM runtime |
| `SPTracking-G1-BFM-WBTeleopActor-HEFTCritic` | deployable WBTeleop `actor` (886-D) | `policy + priv` | BFM | BFM XML/runtime + HEFT observation support |
| `SPTracking-G1-BFM-SPV1Actor-HEFTCritic-HEFTReward` | heading-invariant SPV1 `actor` (1786-D) | `policy + priv` | HEFT | BFM XML/runtime + measured joint-torque sensors |
| `SPTracking-G1-BFM-SPV2Actor-HEFTCritic-HEFTReward` | Compact SPV2: 5-frame history, +4 future, HEFT root rotation (1056-D) | `policy + priv` | HEFT | BFM XML/runtime + measured joint-torque sensors |
| `SPTracking-G1-BFM-SPV3Actor-HEFTCritic-HEFTReward` | SPV3: SPV2 + supervised MLP root-state estimator (6546-D deploy input, 1064-D policy input) | `policy + priv` | HEFT | BFM XML/runtime + measured joint-torque sensors |
| `SPTracking-G1-BFM-SPV4Actor-HEFTCritic-HEFTReward` | SPV4: SPV3 + current root-frame robot/reference/error states for 13 HEFT key bodies (1649-D policy input) | `policy + priv` | HEFT | Privileged BFM simulator body state; not directly deployable |
| `SPTracking-G1-BFM-SPV5Actor-HEFTCritic-HEFTReward` | SPV5: supervised 50-frame noisy qpos encoder + HEFT-style FK into the SPV4 information layout (1649-D policy input) | `policy + priv` | HEFT | Reference side is deployment-compatible; robot key-body state retains SPV4's runtime requirement |
| `SPTracking-G1-BFM-SPV5-1Actor-HEFTCritic-HEFTReward` | SPV5-1: SPV5 + a shared root/contact estimator whose two foot-contact probabilities extend the policy input to 1651-D | `policy + priv` | HEFT | Contact labels are simulation-only; deployment still uses the same 50-frame proprioception and torque history |
| `SPTracking-G1-BFM-SPV5-1MoEActor-HEFTCritic-HEFTReward` | Equal-parameter SPV5-1 residual MoE with 16 experts, top-8 observation-conditioned routing, block-internal LayerNorm, and post-mixture RMSNorm | `policy + priv` | HEFT | No motion/task ID; collect-level load balance and warm-started routing confidence losses |
| `SPTracking-G1-BFM-SPV6Actor-HEFTCritic-HEFTReward` | SPV6: SPV5 + actor-inferred 56-D RMA latent from nominal physics and 50-frame proprioception | HEFT base + actual physics/push latent | HEFT | RMA alignment and privileged reconstruction training |
| `SPTracking-G1-BFM-SPV6-0Actor-HEFTCritic-HEFTReward` | SPV6-0 oracle: SPV5 startup DR + raw actual physics (34-D) + raw 50-frame push window (350-D) | HEFT base + the same raw 384-D oracle input | HEFT | Controlled SPV5 + oracle-information ablation; no DR encoder or reconstruction loss |
| `SPTracking-G1-BFM-SPV6-1Actor-HEFTCritic-HEFTReward` | SPV6-1 oracle: SPV5 + raw actual physics (34-D) + raw 50-frame push window (350-D) | HEFT base + the same raw 384-D oracle input | HEFT | Diagnostic upper bound; no DR encoder, latent alignment, or reconstruction loss |
| `SPTracking-G1-TestPolicyGradients` | SPV5 small-network actor plus rollout-only simple/hard label | `policy + priv` small-network critic | HEFT | Two explicit NPZ files; fixed per-rank 50/50 assignment and minibatch gradient diagnostics |

Every BFM-XML task above except the already-HEFT-reward SPV tasks also has an
additive HEFT-reward variant with the same public naming grammar, for example:

```bash
uv run sp-train task_id=SPTracking-G1-BFM-WBTeleopActor-HEFTCritic-HEFTReward \
  motion_path=/path/to/motions
```

This defines eight additional tasks:

- `SPTracking-G1-BFM-BFMActor-BFMCritic-HEFTReward`
- `SPTracking-G1-BFM-BFMActor-HEFTCritic-HEFTReward`
- `SPTracking-G1-BFM-StudentActor-HEFTCritic-HEFTReward`
- `SPTracking-G1-BFM-TeacherActor-HEFTCritic-HEFTReward`
- `SPTracking-G1-BFM-StudentActor-BFMCritic-HEFTReward`
- `SPTracking-G1-BFM-TeacherActor-BFMCritic-HEFTReward`
- `SPTracking-G1-BFM-WBTeleopActor-BFMCritic-HEFTReward`
- `SPTracking-G1-BFM-WBTeleopActor-HEFTCritic-HEFTReward`

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

SPV3 retains the SPV2 policy contract and extends the six deployable
proprioceptive streams to a shared 50-frame window. A configurable MLP
estimator maps that 6100-D history to pelvis height and three-dimensional root
linear velocity in the current robot frame. The policy reuses only the newest
five frames, then adds the four estimates and their current reference errors,
giving a 1064-D internal policy input. Simulation ground truth is stored only
in the rollout supervision group; it is excluded from actor/critic inputs and
ONNX export. PPO gradients are stopped at the estimator output, so the
estimator is trained exclusively by the separately logged height and linear
velocity MSE losses.

SPV4 is a privileged training variant that additionally observes current
position, 6D rotation, root-relative linear velocity, and root-relative angular
velocity for the 13 HEFT semantic key bodies. Robot and reference states are
expressed in their respective root frames. Reference states are aligned into
the robot root frame before constructing reference-minus-robot errors; rotation
errors use a zero-centered relative 6D rotation. These body states come directly
from the simulator and motion cache, so SPV4 is not a sim-to-real observation
contract until body FK or an equivalent estimator is added.

SPV5 removes the clean-reference side of that contract. Its supervised MLP
reads 50 noisy minimal-qpos frames (`root_pos(3) + root_rot6d(6) +
joint_pos(29)`) at offsets `[-42, +7]` and predicts a normalized residual for
the clean `[-3, +7]` support window. Only `[0, +4]` reaches the policy; the
extra support frames make HEFT's centered differences and five-frame moving
average well-defined at both policy-window boundaries. Reference root
velocity, angular velocity, joint velocity, projected gravity, and all 13
semantic key-body states are rebuilt from the decoded qpos through the BFM G1
kinematic tree. PPO gradients are detached at both the reference encoder and
the SPV3 root estimator. Their separately logged supervised losses are the
only gradients that update those networks.

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
configuration, so `--task-id` is inferred and observation/reward/network
ablations are reconstructed from the checkpoint. Pass `--task-id` only for a
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
