# Motion Observation Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `SP_Tracking` motion-observation math and lifecycle semantics with the reference repository without changing the large-dataset loader or adaptive sampling architecture.

**Architecture:** Keep the current command and loader stack, but replace the observation math in `sp.py` with reference-equivalent frame conventions, buffering semantics, and contact/gravity normalization. Use focused tests to lock down each observation family before changing implementation.

**Tech Stack:** Python 3.13, `pytest`, `torch`, `numpy`, existing `mjlab`/`sp_tracking` env code

---

### Task 1: Lock down joint target and history semantics

**Files:**
- Modify: `tests/test_motion_loader_fps.py`
- Modify: `src/sp_tracking/tasks/tracking/mdp/sp.py`
- Modify: `src/sp_tracking/tasks/tracking/mdp/multi_command_largedataset.py` only if shared-state hooks are needed

- [ ] **Step 1: Write the failing test**

Add a test that checks `target_joint_pos_obs` subtracts the same joint offset convention used by the reference repo, and that `joint_pos_history` / `joint_vel_history` use the same buffer behavior across reset and update.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_motion_loader_fps.py::test_large_dataset_store_prints_motion_chunk_load_progress -v`
Expected: FAIL because joint offset/history alignment is not yet implemented.

- [ ] **Step 3: Write minimal implementation**

Align `target_joint_pos_obs` with the reference pattern:

```python
current_joint_pos = self.asset.data.joint_pos[:, self.joint_idx_asset]
current_joint_pos = current_joint_pos - self.env.action_manager.offset[:, self.joint_idx_asset]
current_joint_pos = current_joint_pos.unsqueeze(1)
target_minus_current = target_joint_pos - current_joint_pos
return torch.cat(
    [target_joint_pos.reshape(self.num_envs, -1),
     target_minus_current.reshape(self.num_envs, -1)],
    dim=-1,
)
```

Move history observations toward reference-style reset/update/compute semantics rather than the current simplified roll-buffer behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_motion_loader_fps.py tests/test_hydra_builders.py tests/test_large_dataset_motion_scan.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sp_tracking/tasks/tracking/mdp/sp.py src/sp_tracking/tasks/tracking/mdp/multi_command_largedataset.py tests/test_motion_loader_fps.py
git commit -m "feat: align joint observation semantics"
```

### Task 2: Align keypoint target frame conventions

**Files:**
- Modify: `tests/test_motion_loader_fps.py`
- Modify: `src/sp_tracking/tasks/tracking/mdp/sp.py`

- [ ] **Step 1: Write the failing test**

Add a controlled-motion test that compares `target_keypoints_pos_b_obs(include_diff=true)` and `target_keypoints_rot_b_obs(include_diff=true)` against the reference frame convention: step-0 root frame as the reference, not a per-step recomputation.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_motion_loader_fps.py::test_large_dataset_store_can_read_metadata_with_process_backend -v`
Expected: FAIL because keypoint frame handling is still using the current semantics.

- [ ] **Step 3: Write minimal implementation**

Rewrite keypoint target observation math to use the same `quat_apply_inverse` / `quat_delta` construction as the reference repository, and preserve `include_diff` packing order.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_motion_loader_fps.py tests/test_hydra_builders.py tests/test_large_dataset_motion_scan.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sp_tracking/tasks/tracking/mdp/sp.py tests/test_motion_loader_fps.py
git commit -m "feat: align keypoint observation frames"
```

### Task 3: Align gravity, contact, and termination hints

**Files:**
- Modify: `tests/test_motion_loader_fps.py`
- Modify: `src/sp_tracking/tasks/tracking/mdp/sp.py`

- [ ] **Step 1: Write the failing test**

Add tests for `projected_gravity_history`, `feet_contact_state`, and motion-derived termination hints to ensure normalization and packing order match the reference implementation.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_motion_loader_fps.py::test_large_dataset_store_writes_and_reuses_json_metadata_cache -v`
Expected: FAIL because gravity/contact math does not yet match the reference implementation.

- [ ] **Step 3: Write minimal implementation**

Normalize projected gravity with the same clamped norm path as the reference implementation and replace hard-coded contact normalization with robot-mass-derived normalization.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_motion_loader_fps.py tests/test_hydra_builders.py tests/test_large_dataset_motion_scan.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sp_tracking/tasks/tracking/mdp/sp.py tests/test_motion_loader_fps.py
git commit -m "feat: align gravity and contact observations"
```

### Task 4: Final verification

**Files:**
- Inspect: `src/sp_tracking/tasks/tracking/mdp/sp.py`
- Inspect: `src/sp_tracking/tasks/tracking/mdp/multi_command_largedataset.py`
- Inspect: `src/sp_tracking/config/build_env.py`
- Inspect: `src/sp_tracking/conf/task/obs/sp_tracking.yaml`

- [ ] **Step 1: Run the full affected test set**

Run: `.venv/bin/python -m pytest tests/test_motion_loader_fps.py tests/test_hydra_builders.py tests/test_large_dataset_motion_scan.py -v`

- [ ] **Step 2: Run a targeted training smoke test**

Run the previously failing `tracking_bfm_sp` training command with the same motion file and confirm the NaN diagnostics no longer point at the aligned observation terms.

- [ ] **Step 3: Commit**

```bash
git add src/sp_tracking/tasks/tracking/mdp/sp.py src/sp_tracking/tasks/tracking/mdp/multi_command_largedataset.py src/sp_tracking/config/build_env.py src/sp_tracking/conf/task/obs/sp_tracking.yaml tests/test_motion_loader_fps.py docs/superpowers/plans/2026-07-10-motion-observation-alignment.md
git commit -m "feat: align motion observation pipeline"
```
