# SP Tracking Hydra Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `SP_Tracking` as an external package that depends on `mjlab==1.5.0`, keeps the current tracking_bfm manager/rsl-rl training behavior, and moves task/agent parameters into Hydra YAML.

**Architecture:** The package owns tracking-specific MDP terms, multi-motion commands, runner/PPO customizations, Hydra configs, task-specific G1 XML assets, and train/play entry points. mjlab remains an external runtime package that provides `ManagerBasedRlEnv`, manager cfg dataclasses, entity/sensor APIs, and rsl-rl wrappers. YAML selects command/observation/reward/termination terms and the robot asset; Python builders translate YAML into mjlab manager cfg objects.

**Tech Stack:** Python 3.10+, uv, Hydra/OmegaConf, mjlab 1.5.0, rsl-rl-lib 5.4.0, pytest.

---

### Task 1: Create External Package Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/sp_tracking/__init__.py`
- Create: `src/sp_tracking/scripts/train.py`
- Create: `tests/test_package_import.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_package_import.py`:

```python
def test_sp_tracking_imports() -> None:
  import sp_tracking

  assert sp_tracking.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_package_import.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing package metadata.

- [ ] **Step 3: Add package metadata and import module**

Create `pyproject.toml` with package name `sp-tracking`, dependencies `mjlab==1.5.0`, `hydra-core`, `omegaconf`, and script `sp-train = sp_tracking.scripts.train:main`.

Create `src/sp_tracking/__init__.py`:

```python
__version__ = "0.1.0"
```

Create a placeholder `src/sp_tracking/scripts/train.py` exposing `main()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_package_import.py -q`
Expected: PASS.

### Task 2: Copy Tracking Code Into External Namespace

**Files:**
- Create: `src/sp_tracking/tasks/tracking/mdp/*.py`
- Create: `src/sp_tracking/tasks/tracking/rl/*.py`
- Create: `src/sp_tracking/tasks/tracking/viewer/*.py`
- Modify copied imports from `mjlab.tasks.tracking` to `sp_tracking.tasks.tracking`.
- Test: `tests/test_tracking_modules_import.py`

- [ ] **Step 1: Write failing module import tests**

Create `tests/test_tracking_modules_import.py`:

```python
def test_tracking_mdp_modules_import() -> None:
  import sp_tracking.tasks.tracking.mdp.multi_commands as multi_commands
  import sp_tracking.tasks.tracking.mdp.multi_command_largedataset as large_dataset
  import sp_tracking.tasks.tracking.mdp.sp as sp_mdp

  assert hasattr(multi_commands, "MotionCommandCfg")
  assert hasattr(large_dataset, "MotionCommandCfg")
  assert hasattr(sp_mdp, "SP_REQUIRED_BODY_NAMES")


def test_tracking_rl_modules_import() -> None:
  from sp_tracking.tasks.tracking.rl import SpTrackingOnPolicyRunner
  from sp_tracking.tasks.tracking.rl.ppo import SparseTrackSplitLrPPO

  assert SpTrackingOnPolicyRunner is not None
  assert SparseTrackSplitLrPPO is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_tracking_modules_import.py -q`
Expected: FAIL because copied modules do not exist.

- [ ] **Step 3: Copy the required modules**

Copy from `tracking_bfm/src/mjlab/tasks/tracking`:

```bash
cp src/mjlab/tasks/tracking/mdp/multi_commands.py SP_Tracking/src/sp_tracking/tasks/tracking/mdp/multi_commands.py
cp src/mjlab/tasks/tracking/mdp/multi_command_largedataset.py SP_Tracking/src/sp_tracking/tasks/tracking/mdp/multi_command_largedataset.py
cp src/mjlab/tasks/tracking/mdp/observations.py SP_Tracking/src/sp_tracking/tasks/tracking/mdp/observations.py
cp src/mjlab/tasks/tracking/mdp/rewards.py SP_Tracking/src/sp_tracking/tasks/tracking/mdp/rewards.py
cp src/mjlab/tasks/tracking/mdp/terminations.py SP_Tracking/src/sp_tracking/tasks/tracking/mdp/terminations.py
cp src/mjlab/tasks/tracking/mdp/metrics.py SP_Tracking/src/sp_tracking/tasks/tracking/mdp/metrics.py
cp src/mjlab/tasks/tracking/mdp/sp.py SP_Tracking/src/sp_tracking/tasks/tracking/mdp/sp.py
cp src/mjlab/tasks/tracking/rl/runner.py SP_Tracking/src/sp_tracking/tasks/tracking/rl/runner.py
cp src/mjlab/tasks/tracking/rl/ppo.py SP_Tracking/src/sp_tracking/tasks/tracking/rl/ppo.py
cp src/mjlab/tasks/tracking/rl/attention_models.py SP_Tracking/src/sp_tracking/tasks/tracking/rl/attention_models.py
```

Add `__init__.py` files and rewrite imports to the `sp_tracking.tasks.tracking` namespace.

- [ ] **Step 4: Run tests to verify imports pass**

Run: `PYTHONPATH=src pytest tests/test_tracking_modules_import.py -q`
Expected: PASS.

### Task 3: Add Hydra YAML And Builder

**Files:**
- Create: `src/sp_tracking/conf/train.yaml`
- Create: `src/sp_tracking/conf/task/tracking_bfm.yaml`
- Create: `src/sp_tracking/conf/task/tracking_bfm_sp.yaml`
- Create: `src/sp_tracking/conf/task/command/multimotion.yaml`
- Create: `src/sp_tracking/conf/task/command/largedataset.yaml`
- Create: `src/sp_tracking/conf/task/obs/old_tracking.yaml`
- Create: `src/sp_tracking/conf/task/obs/sp_tracking.yaml`
- Create: `src/sp_tracking/conf/task/reward/old_tracking.yaml`
- Create: `src/sp_tracking/conf/task/reward/sp_tracking.yaml`
- Create: `src/sp_tracking/conf/agent/tracking_bfm_ppo.yaml`
- Create: `src/sp_tracking/config/build_env.py`
- Create: `src/sp_tracking/config/build_agent.py`
- Create: `tests/test_hydra_builders.py`

- [ ] **Step 1: Write failing builder tests**

Create tests that compose Hydra configs and assert:
- default `tracking_bfm` uses `MultiMotionCommandCfg`;
- `tracking_bfm_sp` uses large dataset command;
- actor/critic observation groups exist;
- reward and termination names match the current defaults.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_hydra_builders.py -q`
Expected: FAIL because builders/configs do not exist.

- [ ] **Step 3: Implement YAML configs and builders**

Implement registry dictionaries mapping YAML term names to copied `mdp` functions and mjlab common `mdp` functions. Builders must construct `ManagerBasedRlEnvCfg`, `ObservationGroupCfg`, `ObservationTermCfg`, `RewardTermCfg`, `TerminationTermCfg`, command cfg, events, action cfg, scene cfg, sensors, viewer, and sim cfg.

- [ ] **Step 4: Run builder tests**

Run: `PYTHONPATH=src pytest tests/test_hydra_builders.py -q`
Expected: PASS.

### Task 4: Add Hydra RSL-RL Training Entry

**Files:**
- Modify: `src/sp_tracking/scripts/train.py`
- Create: `tests/test_train_entry.py`

- [ ] **Step 1: Write failing train-entry tests**

Create tests that call a pure helper such as `build_training_objects(cfg)` or `prepare_train_cfg(cfg)` without constructing a live MuJoCo env. Assert that Hydra overrides update env and agent cfg values.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_train_entry.py -q`
Expected: FAIL because train helpers are missing.

- [ ] **Step 3: Implement the train entry**

Use Hydra `@hydra.main(config_path="../conf", config_name="train")`. The entry composes env/agent configs, applies motion path overrides, creates `ManagerBasedRlEnv`, wraps with `RslRlVecEnvWrapper`, creates `SpTrackingOnPolicyRunner`, handles resume/debug/log dir, and calls `runner.learn()`.

- [ ] **Step 4: Run train-entry tests**

Run: `PYTHONPATH=src pytest tests/test_train_entry.py -q`
Expected: PASS.

### Task 5: Verify, Move, Initialize Git

**Files:**
- All project files.

- [ ] **Step 1: Run smoke tests**

Run: `PYTHONPATH=src pytest tests/test_package_import.py tests/test_tracking_modules_import.py tests/test_hydra_builders.py tests/test_train_entry.py -q`
Expected: PASS or documented dependency blocker.

- [ ] **Step 2: Move project to final path**

Run with filesystem approval:

```bash
mv /home/lenovo/workspace/UNICTL/tracking_bfm/SP_Tracking /home/lenovo/workspace/UNICTL/SP_Tracking
```

- [ ] **Step 3: Initialize git and commit**

Run:

```bash
git init
git add .
git commit -m "Initial SP Tracking extraction"
git remote add origin git@github.com:mwondering/SP_Tracking.git
```

- [ ] **Step 4: Verify repository state**

Run: `git status --short` in `/home/lenovo/workspace/UNICTL/SP_Tracking`
Expected: clean worktree.
