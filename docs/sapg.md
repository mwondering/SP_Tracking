# SAPG-PPO

The repository includes an opt-in implementation of Split and Aggregate Policy
Gradients (SAPG), following the behavior of the authors' official
`rl_games`-based implementation. It is an extension of the existing PPO
algorithms rather than a separate task or model family.

## Enabling SAPG

The same switch applies to every registered tracking task:

```bash
uv run sp-train \
  task_id=SPTracking-G1-BFM-BFMActor-BFMCritic \
  motion_path=/path/to/motions \
  task.num_envs=16 \
  agent.algorithm.sapg_cfg.enabled=true
```

For a direct Hydra task selection:

```bash
uv run sp-train \
  task=tracking_bfm_spv6_actor_heft_critic_heft_reward \
  motion_path=/path/to/motions \
  agent.algorithm.sapg_cfg.enabled=true
```

The default configuration is:

```yaml
sapg_cfg:
  enabled: false
  compatibility: official
  num_policy_blocks: 4
  local_parameter_dim: 32
  off_policy_ratio: 1
  exploration_type: none
  entropy_coef_scale: 1.0
  value_eval_chunk_size: 8192
```

`task.num_envs` is the per-rank environment count and must be divisible by
`num_policy_blocks` during training. Play configurations may use one environment;
when there is no explicit SAPG policy context, inference always selects the last
block (the leader).

## Compatibility contract

SAPG is disabled by default. When disabled:

- plain agents retain `algorithm.class_name: PPO`;
- `sapg_cfg` is removed before serialization into the RSL-RL constructor;
- actor, critic, distribution, rollout storage and optimizer construction are
  unchanged;
- the original PPO/SPV/HEFT `act`, `compute_returns`, `update`, checkpoint and
  export branches are used.

When enabled, the builder rewrites only a plain `PPO` to the local tracking PPO
extension. Existing `HeftTeacherPPO`, `SPV3EstimatorPPO`,
`SPV5ReferenceEncoderPPO`, `SPV51ContactEstimatorPPO` and `SPV6RmaPPO` classes
are retained and receive the same SAPG extension.

## Official-code semantics

- Environments are split into contiguous policy blocks.
- The final block is the leader; preceding blocks are followers.
- Actor and critic share a learned local policy embedding (32 dimensions by
  default), injected at the first layer of their final MLP heads.
- Each policy block has its own state-independent Gaussian standard deviation.
- Every update retains all original on-policy trajectories and randomly chooses
  `off_policy_ratio` follower blocks whose trajectories are additionally trained
  as leader data.
- Aggregation preserves complete environment trajectories and uses the original
  base PPO mini-batch size. As in the official dataset, any remainder is folded
  into the final mini-batch instead of creating an extra optimizer step.
- Aggregated actions and old log probabilities remain frozen follower behavior
  data for the PPO importance ratio. Current policy/value evaluation uses the
  leader.
- Distribution parameters used by adaptive KL start from the follower behavior
  distribution and are refreshed per aggregate sample after every mini-batch,
  matching the official dataset's `update_mu_sigma` behavior across PPO epochs.
- The aggregated return is the official one-step target
  `stored_reward + gamma * alive * V_leader(next_state)`.
- PPO uses the normal behavior-policy ratio and `[1-eps, 1+eps]` clipping, as in
  the official source code.
- On-policy returns remain GAE.

`exploration_type: none` retains the task's existing entropy coefficient.
`exploration_type: entropy` uses the official per-policy coefficients linearly
spaced from `0.5` to `0.0`, multiplied by `entropy_coef_scale`; the leader then
has zero entropy reward.

SPV supervised/representation losses are evaluated only on original on-policy
samples and weighted as masked full-batch means, so duplicated follower
trajectories do not increase their epoch weight.

## Checkpoints and deployment

SAPG checkpoints store the policy embedding, per-block standard deviations,
conditioned actor/critic weights, optimizer state, configuration and follower
selection RNG state. Checkpoints with the same SAPG configuration load strictly.

A base PPO checkpoint can initialize SAPG actor and critic weights with
`strict=false` and optimizer loading disabled. Its one-dimensional standard
deviation is copied to every SAPG block.

Actor export deep-copies the policy head and folds the leader embedding into the
first-layer bias. The exported JIT/ONNX policy therefore has the same inputs as
the original task and contains no SAPG policy-ID input.

## Current limits

Official compatibility currently rejects recurrent policies, RND,
non-`GaussianDistribution` actors and `torch_compile_mode`. All currently
registered tracking tasks use feed-forward Gaussian policies and are supported.
