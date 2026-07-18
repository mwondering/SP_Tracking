# Two-motion policy-gradient diagnostics

`SPTracking-G1-TestPolicyGradients` is a training-only SPV5 experiment for
comparing one simple motion, one hard motion, and their fixed 50/50 mixture.
It uses 16,384 environments **per rank/GPU** and trains every model from
scratch with `[512, 256, 128]` actor, critic, estimator, and reference-encoder
hidden dimensions.

## Three runs

Use the same seed for the first paired comparison and distinct run names. Do
not set `agent.resume=true` or pass a checkpoint.

The repository launcher starts all three conditions concurrently on separate
GPU groups:

```bash
scripts/train_test_policy_gradients.sh \
  /absolute/path/simple.npz \
  /absolute/path/hard.npz \
  agent.run_name=gradient_mixed \
  agent.seed=42
```

It defaults to `SP_TRACKING_GRADIENT_GPU_GROUPS="0;1;2"`. For two GPUs per
condition, use `SP_TRACKING_GRADIENT_GPU_GROUPS="0,1;2,3;4,5"`. The launcher
also accepts the Hydra arguments copied from one of the commands below,
replaces `mode` for each child, and derives `_simple`, `_hard`, and `_mixed`
run-name suffixes. Set `SP_TRACKING_DRY_RUN=1` to print the three commands.

The equivalent individual commands are:

```bash
uv run sp-train \
  task_id=SPTracking-G1-TestPolicyGradients \
  task.gradient_test.mode=simple \
  task.gradient_test.simple_motion_file=/absolute/path/simple.npz \
  agent.run_name=gradient_simple \
  seed=42 task.seed=42 agent.seed=42

uv run sp-train \
  task_id=SPTracking-G1-TestPolicyGradients \
  task.gradient_test.mode=hard \
  task.gradient_test.hard_motion_file=/absolute/path/hard.npz \
  agent.run_name=gradient_hard \
  seed=42 task.seed=42 agent.seed=42

uv run sp-train \
  task_id=SPTracking-G1-TestPolicyGradients \
  task.gradient_test.mode=mixed \
  task.gradient_test.simple_motion_file=/absolute/path/simple.npz \
  task.gradient_test.hard_motion_file=/absolute/path/hard.npz \
  agent.run_name=gradient_mixed \
  seed=42 task.seed=42 agent.seed=42
```

For multi-GPU training, pass the same overrides through the repository's
`torchrun` launcher. Each rank loads both archives in mixed mode and assigns
its first 8,192 environments to the simple motion and its remaining 8,192 to
the hard motion. The assignment is based on environment ID and survives every
reset. Single-motion runs only require the file they use.

The default rollout has 24 steps, 16 minibatches, and five PPO learning epochs,
so the logger writes 80 raw optimizer-minibatch records per iteration. Mixed
minibatches are exactly stratified: on each rank, every minibatch contains the
same number of simple and hard samples.

## Gradient definition

Diagnostics are measured after PPO advantage normalization and before the real
training backward pass, gradient clipping, and optimizer step.

- Actor: the actual PPO policy objective (clipped surrogate minus the weighted
  entropy term) with respect to the main SPV5 policy MLP and trainable action
  distribution parameters. The detached SPV5 root estimator and reference
  encoder keep their existing supervised update paths and MSE logs.
- Critic: the weighted value-loss gradient with respect to every critic
  parameter.
- Multi-GPU: each motion gradient is weighted by its local sample count,
  summed across ranks, and divided by the global sample count before norms,
  dot products, and cosines are computed.

For mixed runs, the most direct conflict indicators are
`actor_grad_cosine`, `critic_grad_cosine`, and `*_grad_cancellation`.
Cosine below zero indicates opposing directions. Cancellation approaches zero
when equal-magnitude task gradients cancel, and one when they align. Norm
ratios quantify scale imbalance. Per-motion advantage, return, loss, entropy,
KL, clip fraction, and mean phase are stored alongside them to help distinguish
optimization conflict from unequal data or episode-phase distributions.

## Logs and comparisons

Each run directory contains `gradient_diagnostics.sqlite`. Its
`minibatch_gradients` table stores one globally aggregated raw row for every
optimizer minibatch. `cfg.yaml` in the same directory records the selected
motion path(s), mode, seed, and training configuration. W&B and terminal output
receive only per-iteration aggregates; W&B tags use the
`GradientDiagnostics/<metric>/{mean,std,min,max}` hierarchy.

The three runs answer two slightly different questions:

1. **Compute-matched:** compare all runs at the same iteration. This tests the
   practical effect of sharing a fixed total training budget.
2. **Per-motion-exposure-matched:** also compare a mixed checkpoint at twice
   the single-motion iteration, because each motion receives half of the mixed
   samples. This separates gradient interference from simple data dilution.

After a pilot with one shared seed, repeat all three conditions with several
paired seeds. Report the distribution across seeds and aggregate minibatch
statistics over a fixed iteration window rather than selecting a single
minibatch or checkpoint.
