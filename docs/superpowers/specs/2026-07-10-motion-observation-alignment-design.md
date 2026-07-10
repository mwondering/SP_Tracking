# Motion Observation Alignment Design

## Background

`SP_Tracking` currently computes motion-tracking observations through `src/sp_tracking/tasks/tracking/mdp/sp.py`, while the reference repository `/home/lenovo/workspace/UNICTL/motion_tracking` uses a different observation chain in `active_adaptation/envs/mdp/commands/motion_tracking/observations.py` and `active_adaptation/envs/mdp/observations/core.py`.

Recent NaN failures point to observation computation rather than dataset scanning or metadata loading. The goal is to align the observation chain with the reference implementation without replacing the large-dataset loader, active-subset cache, or adaptive sampling framework.

## Goal

Fully align the motion-observation pipeline in `SP_Tracking` with the semantics used in `motion_tracking`, while keeping the current large-dataset architecture intact.

Success criteria:

1. Motion-derived observation terms follow the same math and frame conventions as the reference repository.
2. History observations use reference-compatible reset, update, and compute semantics.
3. Joint observations use the same offset and shared-state conventions as the reference repository.
4. Keypoint, gravity, and contact observations match the reference repository's numerical behavior.
5. Existing large-dataset startup, manifest/cache reuse, and active-subset loading behavior remain unchanged.

## In Scope

Files expected to change:

1. `src/sp_tracking/tasks/tracking/mdp/sp.py`
2. `src/sp_tracking/tasks/tracking/mdp/multi_command_largedataset.py` only where observation parity needs additional shared-state hooks
3. `src/sp_tracking/config/build_env.py` if observation wiring requires extra bindings
4. `src/sp_tracking/conf/task/obs/sp_tracking.yaml` if default observation configuration must be aligned
5. Targeted tests under `tests/`

## Out of Scope

1. Rewriting the large-dataset loader or active-subset machinery
2. Changing reward logic unless required by observation interface parity
3. Changing PPO, runner, distributed launch, or cache generation logic
4. Porting the entire `motion_tracking` command system wholesale

## Alignment Targets

The following observation terms or behaviors must match the reference implementation:

1. `command_obs`
2. `target_joint_pos_obs`
3. `target_root_z_obs`
4. `target_projected_gravity_b_obs`
5. `root_angvel_b_history`
6. `root_linvel_b_history`
7. `projected_gravity_history`
8. `joint_pos_history`
9. `joint_vel_history`
10. `current_keypoint_pos_b_obs`
11. `current_keypoint_rot_b_obs`
12. `current_keypoint_linvel_b_obs`
13. `current_keypoint_angvel_b_obs`
14. `target_keypoints_pos_b_obs`
15. `target_keypoints_rot_b_obs`
16. `feet_contact_state`
17. `target_feet_contact_state_obs`
18. `body_z_termination_obs` where it depends on motion-derived state

## Key Differences Identified

### Joint target and history semantics

The reference repository subtracts action-manager joint offsets and uses shared noisy joint-state pathways for some observation terms. `SP_Tracking` currently computes these terms more directly from robot state. That means `target_joint_pos_obs`, `joint_pos_history`, and `joint_vel_history` are not semantically aligned.

### Keypoint target frame conventions

The reference repository defines target keypoint body-frame observations relative to the reference motion's step-0 root frame. `SP_Tracking` currently recomputes target keypoints in a different way for future steps. This affects `target_keypoints_pos_b_obs` and `target_keypoints_rot_b_obs`.

### Stateful observation lifecycle

The reference repository separates reset, update, and compute phases for stateful observations through ring-buffer-like utilities. `SP_Tracking` currently uses simplified rolling buffers, which changes the effective meaning of history observations.

### Gravity and contact normalization

The reference repository explicitly normalizes projected gravity and derives contact normalization from robot mass. `SP_Tracking` currently has a lighter gravity path and a hard-coded contact-force denominator. These are behavior mismatches even when they are not the immediate NaN source.

## Design

Implement full observation-chain parity term by term inside `SP_Tracking` without changing the loader architecture.

Design rules:

1. Keep `LargeDatasetMultiMotionCommand`, active subset loading, and motion gather APIs as the source of motion samples.
2. Refactor observation math in `sp.py` so each term matches the reference implementation's frame and buffer semantics.
3. Add only the minimal missing shared-state hooks needed for parity, instead of replacing the entire command system.
4. Preserve external observation term names and config structure whenever possible so training configs remain stable.

## Implementation Phases

### Phase 1: Shared-state parity

1. Add any missing shared joint-position and joint-velocity access paths needed by the reference observation semantics.
2. Align action-offset handling for joint observations.
3. Keep the helper boundaries explicit so parity logic stays testable.

### Phase 2: Stateful observation parity

1. Rework history observations to use reference-compatible reset, update, and compute behavior.
2. Align `root_angvel_b_history`, `root_linvel_b_history`, `projected_gravity_history`, `joint_pos_history`, and `joint_vel_history`.

### Phase 3: Motion-target observation parity

1. Align `command_obs`.
2. Align `target_joint_pos_obs`, `target_root_z_obs`, and `target_projected_gravity_b_obs`.
3. Align current-keypoint and target-keypoint observations with the reference frame conventions used in `motion_tracking`.
4. Align `feet_contact_state`, `target_feet_contact_state_obs`, and motion-derived termination-hint observations.

### Phase 4: Diagnostics and validation

1. Add targeted tests for joint-target alignment, history-buffer semantics, keypoint frame semantics, and contact normalization.
2. Add a regression test path for the previously failing NaN scenario as far as local test fixtures allow.
3. Keep the existing non-finite diagnostics in the runner and verify they still provide useful motion context after alignment.

## Validation Plan

Minimum validation before completion:

1. Existing observation-related tests still pass.
2. New tests cover the aligned terms and stateful observation semantics.
3. A targeted rerun on the failing training setup no longer produces NaN observations for the same motion context, or any remaining failure is shown to come from a different layer with fresh diagnostics.
4. Large-dataset startup and cache behavior remain unchanged.

## Recommendation

Proceed with full observation-chain alignment in `SP_Tracking`, but keep the large-dataset loader, active-subset framework, and training runner untouched. The safest path is term-by-term parity with focused tests, not a wholesale subsystem transplant.
