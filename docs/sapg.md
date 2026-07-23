# PPO / SAPG / CPO

The repository exposes three policy-optimization variants over the same task
and model profiles:

- `ppo`: the unchanged upstream/tracking PPO path;
- `sapg`: Split and Aggregate Policy Gradients, following the authors'
  official `rl_games` implementation;
- `cpo`: Coupled Policy Optimization, extending the SAPG leader/follower
  aggregate with the official follower PPO-KL and AWAC objectives.

SAPG and CPO are extensions of the existing PPO classes rather than separate
task or model families.

## Selecting the algorithm

PPO is the default. The same selector applies to every registered tracking
task:

```bash
# Plain PPO
uv run sp-train \
  task_id=SPTracking-G1-BFM-BFMActor-BFMCritic \
  motion_path=/path/to/motions \
  agent.algorithm.variant=ppo

# SAPG
uv run sp-train \
  task_id=SPTracking-G1-BFM-BFMActor-BFMCritic \
  motion_path=/path/to/motions \
  task.num_envs=16 \
  agent.algorithm.variant=sapg

# CPO
uv run sp-train \
  task_id=SPTracking-G1-BFM-BFMActor-BFMCritic \
  motion_path=/path/to/motions \
  task.num_envs=16 \
  agent.algorithm.variant=cpo
```

For a direct Hydra task selection:

```bash
uv run sp-train \
  task=tracking_bfm_spv6_actor_heft_critic_heft_reward \
  motion_path=/path/to/motions \
  agent.algorithm.variant=cpo
```

The legacy `agent.algorithm.sapg_cfg.enabled=true` switch remains supported and
selects SAPG. New experiments should use `algorithm.variant`, because it makes
the PPO/SAPG/CPO choice explicit.

The shared ensemble configuration is:

```yaml
variant: ppo
sapg_cfg:
  enabled: false
  method: sapg
  compatibility: official
  num_policy_blocks: 4
  local_parameter_dim: 32
  off_policy_ratio: 1
  exploration_type: none
  entropy_coef_scale: 1.0
  value_eval_chunk_size: 8192
  cpo_awac_temperature: 0.2
  cpo_awac_max_weight: 100.0
  cpo_awac_coef: 0.001
  cpo_kl_coef: 0.0
```

`task.num_envs` is the per-rank environment count and must be divisible by
`num_policy_blocks` for SAPG and CPO training. Play configurations may use one
environment; when there is no explicit policy context, inference always selects
the last block (the leader).

## Compatibility contract

The ensemble extension is disabled for `variant: ppo`. In that mode:

- plain agents retain `algorithm.class_name: PPO`;
- `sapg_cfg` is removed before serialization into the RSL-RL constructor;
- actor, critic, distribution, rollout storage and optimizer construction are
  unchanged;
- the original PPO/SPV/HEFT `act`, `compute_returns`, `update`, checkpoint and
  export branches are used.

For SAPG or CPO, the builder rewrites only a plain `PPO` to the local tracking
PPO extension. Existing `HeftTeacherPPO`, `SPV3EstimatorPPO`,
`SPV5ReferenceEncoderPPO`, `SPV51ContactEstimatorPPO` and `SPV6RmaPPO` classes
are retained and receive the same ensemble extension.

## Shared SAPG semantics

- Environments are split into contiguous policy blocks.
- The final block is the leader; preceding blocks are followers.
- Actor and critic share a learned local policy embedding, injected at the
  first layer of their final MLP heads.
- Each block has an independent state-independent Gaussian standard deviation.
- Every update retains all original on-policy trajectories and randomly chooses
  `off_policy_ratio` follower blocks whose trajectories are additionally trained
  as leader data.
- Aggregation preserves complete trajectories and the base PPO mini-batch size.
  Any remainder is folded into the final mini-batch.
- Aggregated actions and old log probabilities remain frozen follower behavior
  data; current policy/value evaluation uses the leader.
- Adaptive-KL distribution references start from the behavior distribution and
  are refreshed after each mini-batch while behavior log probabilities remain
  immutable.
- The aggregated return is the official one-step target
  `stored_reward + gamma * alive * V_leader(next_state)`.
- Original on-policy returns remain GAE.

`exploration_type: none` retains the task's entropy coefficient.
`exploration_type: entropy` uses per-policy coefficients linearly spaced from
`0.5` to `0.0`, multiplied by `entropy_coef_scale`; the leader then has zero
entropy reward.

SPV supervised and representation losses are evaluated only on original
on-policy samples and weighted as masked full-batch means. Duplicated ensemble
trajectories therefore do not increase their epoch weight.

## CPO semantics

CPO retains all shared SAPG behavior and adds two follower-side roles:

- original follower trajectories use PPO with an optional sampled
  leader/follower log-probability distance term (`cpo_kl_coef`);
- every leader trajectory is re-evaluated under every follower and contributes
  an advantage-weighted behavior-cloning loss (`cpo_awac_*`).

The leader-to-follower copies affect only the actor AWAC objective. They are
excluded from critic losses and all SPV/HEFT auxiliary objectives. Their values
and one-step targets are evaluated under the target follower, so the AWAC weight
uses the corresponding follower advantage.

`cpo_awac_temperature` controls the exponential advantage weight,
`cpo_awac_max_weight` clips it, and `cpo_awac_coef` scales the AWAC loss. The
checked-in defaults use the official implementation's common temperature and
clipping values, with a non-zero AWAC coefficient so selecting CPO actually
couples followers to the leader. `cpo_kl_coef` defaults to zero, matching the
released task configurations; it enables the additional sampled PPO-KL term
when set above zero.

The CPO update reports:

- `cpo/leader_ppo_loss`, `cpo/follower_ppo_kl_loss`, and `cpo/awac_loss`;
- `cpo/follower_to_leader_kl`;
- `cpo/importance_ratio_abs_deviation`;
- `cpo/importance_ess_normalized`;
- clipping, selected-follower, and per-block standard-deviation diagnostics
  under the `cpo/` namespace.

The paper's adversarial discriminator reward is task-dependent and is not yet
implemented here. The current `cpo` variant is therefore the paper's core
KL/AWAC coupling path, not the optional full-CPO discriminator configuration.

## Checkpoints and deployment

SAPG/CPO checkpoints store the method, policy embedding, per-block standard
deviations, conditioned actor/critic weights, optimizer state, configuration
and follower-selection RNG state. A CPO checkpoint cannot be silently resumed
as SAPG, or vice versa.

A base PPO checkpoint can initialize SAPG or CPO actor and critic weights with
`strict=false` and optimizer loading disabled. Its one-dimensional standard
deviation is copied to every policy block.

Actor export deep-copies the policy head and folds the leader embedding into the
first-layer bias. Exported JIT/ONNX policies have the original task inputs and
contain no training-time policy-ID input.

## Current limits

Official compatibility currently rejects recurrent policies, RND,
non-`GaussianDistribution` actors and `torch_compile_mode`. All currently
registered tracking tasks use feed-forward Gaussian policies and are supported.
CPO currently omits the optional adversarial discriminator reward.
