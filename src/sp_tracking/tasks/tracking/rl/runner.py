import json
import os
import time
from pathlib import Path
from typing import cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.utils import check_nan
from rsl_rl.utils.log_writer import LogWriter

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from sp_tracking.tasks.tracking.mdp import MotionCommand
from sp_tracking.tasks.tracking.rl.export import export_sim2real_policy_onnx


def _bootstrap_debug(message: str) -> None:
  debug_dir = os.environ.get("MJLAB_BOOTSTRAP_DEBUG_DIR", "")
  if not debug_dir:
    return
  rank = os.environ.get("RANK", "unknown")
  local_rank = os.environ.get("LOCAL_RANK", "unknown")
  pid = os.getpid()
  line = (
    f"[BOOT][{time.strftime('%Y-%m-%d %H:%M:%S')}] "
    f"rank={rank} local_rank={local_rank} pid={pid}: tracking_runner: {message}"
  )
  print(line, flush=True)
  try:
    os.makedirs(debug_dir, exist_ok=True)
    log_file = os.path.join(debug_dir, f"rank_{rank}_local_{local_rank}_pid_{pid}.log")
    with open(log_file, "a", encoding="utf-8") as f:
      f.write(line + "\n")
      f.flush()
  except Exception:
    pass


def _upload_launch_script_artifact(
  logger: object, launch_script_artifact_path: str | Path | None
) -> bool:
  if launch_script_artifact_path is None:
    return False

  writer = getattr(logger, "writer", None)
  if not isinstance(writer, LogWriter):
    return False

  writer.save_file(str(launch_script_artifact_path))
  return True


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
    launch_script_artifact_path: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name
    self.launch_script_artifact_path = launch_script_artifact_path
    self._launch_script_artifact_uploaded = False

  def _upload_launch_script_artifact_once(self) -> None:
    if self._launch_script_artifact_uploaded:
      return
    self._launch_script_artifact_uploaded = _upload_launch_script_artifact(
      self.logger, self.launch_script_artifact_path
    )

  def _has_motion_tracking_curriculum(self) -> bool:
    unwrapped_env = getattr(self.env, "unwrapped", None)
    curriculum_manager = getattr(unwrapped_env, "curriculum_manager", None)
    active_terms = getattr(curriculum_manager, "active_terms", ())
    return "motion_tracking_progress" in active_terms

  @staticmethod
  def _record_schedule_state(
    state: dict[str, float], prefix: str, name: str, result: object
  ) -> None:
    if not isinstance(result, dict):
      return
    for key, value in result.items():
      try:
        state[f"{prefix}/{name}/{key}"] = float(value)
      except (TypeError, ValueError):
        continue

  def _step_motion_tracking_curriculum(self, iteration: int) -> None:
    if not self._has_motion_tracking_curriculum():
      return
    max_iterations = max(int(self.cfg.get("max_iterations", iteration + 1)), 1)
    progress = min(max(float(iteration + 1) / float(max_iterations), 0.0), 1.0)
    unwrapped_env = self.env.unwrapped
    state: dict[str, float] = {"progress": progress}

    command_manager = getattr(unwrapped_env, "command_manager", None)
    for name in getattr(command_manager, "active_terms", ()):
      term = command_manager.get_term(name)
      step_schedule = getattr(term, "step_schedule", None)
      if callable(step_schedule):
        result = step_schedule(progress, iteration)
        self._record_schedule_state(state, "command", str(name), result)

    action_manager = getattr(unwrapped_env, "action_manager", None)
    for name in getattr(action_manager, "active_terms", ()):
      term = action_manager.get_term(name)
      step_schedule = getattr(term, "step_schedule", None)
      if callable(step_schedule):
        result = step_schedule(progress, iteration)
        self._record_schedule_state(state, "action", str(name), result)

    event_manager = getattr(unwrapped_env, "event_manager", None)
    for names in getattr(event_manager, "active_terms", {}).values():
      for name in names:
        term = event_manager.get_term_cfg(name).func
        step_schedule = getattr(term, "step_schedule", None)
        if callable(step_schedule):
          result = step_schedule(progress, iteration)
          self._record_schedule_state(state, "event", str(name), result)

    reward_manager = getattr(unwrapped_env, "reward_manager", None)
    for name in getattr(reward_manager, "active_terms", ()):
      term = reward_manager.get_term_cfg(name).func
      step_schedule = getattr(term, "step_schedule", None)
      if callable(step_schedule):
        result = step_schedule(progress, iteration)
        self._record_schedule_state(state, "reward", str(name), result)

    unwrapped_env._motion_tracking_curriculum_state = state

  def _begin_adaptive_sampling_iteration(self, iteration: int) -> None:
    _bootstrap_debug(f"before begin_adaptive_sampling_iteration iteration={iteration}")
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    begin_iteration = getattr(motion_cmd, "begin_adaptive_sampling_iteration", None)
    if callable(begin_iteration):
      begin_iteration(iteration)
    _bootstrap_debug(f"after begin_adaptive_sampling_iteration iteration={iteration}")

  def _write_large_dataset_snapshot(self, iteration: int) -> None:
    unwrapped_env = getattr(self.env, "unwrapped", None)
    command_manager = getattr(unwrapped_env, "command_manager", None)
    if command_manager is None:
      return
    motion_cmd = command_manager.get_term("motion")
    write_snapshot = getattr(motion_cmd, "maybe_write_adaptive_bin_snapshot", None)
    if not callable(write_snapshot):
      return
    log_dir = getattr(self.logger, "log_dir", None) or self.log_dir
    default_snapshot_dir = (
      os.path.join(log_dir, "adaptive_bin_pool_view") if log_dir else None
    )
    write_snapshot(
      iteration=iteration,
      default_snapshot_dir=default_snapshot_dir,
    )

  def _log_large_dataset_timing(
    self, *, it: int, collect_time: float, learn_time: float
  ) -> None:
    unwrapped_env = getattr(self.env, "unwrapped", None)
    command_manager = getattr(unwrapped_env, "command_manager", None)
    if command_manager is None:
      return
    motion_cmd = command_manager.get_term("motion")
    get_stats = getattr(motion_cmd, "get_large_dataset_timing_stats", None)
    if not callable(get_stats):
      return

    try:
      stats = get_stats(reset=True)
    except TypeError:
      stats = get_stats()
    global_bin_update_time = float(stats.get("global_bin_update_time", 0.0))
    global_bin_update_pack_time = float(
      stats.get("global_bin_update_pack_time", 0.0)
    )
    global_bin_update_gather_time = float(
      stats.get("global_bin_update_gather_time", 0.0)
    )
    global_bin_update_apply_time = float(
      stats.get("global_bin_update_apply_time", 0.0)
    )
    adaptive_bin_pool_reset_time = float(
      stats.get("adaptive_bin_pool_reset_time", 0.0)
    )
    adaptive_bin_pool_reset_applied = float(
      stats.get("adaptive_bin_pool_reset_applied", 0.0)
    )
    global_bin_update_episode_key_count = float(
      stats.get("global_bin_update_episode_key_count", 0.0)
    )
    global_bin_update_failure_key_count = float(
      stats.get("global_bin_update_failure_key_count", 0.0)
    )
    subset_update_time = float(stats.get("subset_update_time", 0.0))
    motion_gather_time = float(stats.get("motion_gather_time", 0.0))
    motion_gather_call_count = float(stats.get("motion_gather_call_count", 0.0))
    print(
      "Large dataset timing: "
      f"collect_time: {collect_time:.4f}s, "
      f"learn_time: {learn_time:.4f}s, "
      f"global_bin_update_time: {global_bin_update_time:.4f}s, "
      f"global_bin_update_pack_time: {global_bin_update_pack_time:.4f}s, "
      f"global_bin_update_gather_time: {global_bin_update_gather_time:.4f}s, "
      f"global_bin_update_apply_time: {global_bin_update_apply_time:.4f}s, "
      f"adaptive_bin_pool_reset_time: {adaptive_bin_pool_reset_time:.4f}s, "
      f"adaptive_bin_pool_reset_applied: {adaptive_bin_pool_reset_applied:.0f}, "
      f"global_bin_update_episode_key_count: {global_bin_update_episode_key_count:.0f}, "
      f"global_bin_update_failure_key_count: {global_bin_update_failure_key_count:.0f}, "
      f"subset_update_time: {subset_update_time:.4f}s, "
      f"motion_gather_time: {motion_gather_time:.4f}s, "
      f"motion_gather_call_count: {motion_gather_call_count:.0f}"
    )
    writer = getattr(self.logger, "writer", None)
    if writer is not None:
      writer.add_scalar("Perf/global_bin_update_time", global_bin_update_time, it)
      writer.add_scalar(
        "Perf/global_bin_update_pack_time", global_bin_update_pack_time, it
      )
      writer.add_scalar(
        "Perf/global_bin_update_gather_time", global_bin_update_gather_time, it
      )
      writer.add_scalar(
        "Perf/global_bin_update_apply_time", global_bin_update_apply_time, it
      )
      writer.add_scalar(
        "Perf/adaptive_bin_pool_reset_time", adaptive_bin_pool_reset_time, it
      )
      writer.add_scalar(
        "Perf/adaptive_bin_pool_reset_applied",
        adaptive_bin_pool_reset_applied,
        it,
      )
      writer.add_scalar(
        "Perf/global_bin_update_episode_key_count",
        global_bin_update_episode_key_count,
        it,
      )
      writer.add_scalar(
        "Perf/global_bin_update_failure_key_count",
        global_bin_update_failure_key_count,
        it,
      )
      writer.add_scalar("Perf/subset_update_time", subset_update_time, it)
      writer.add_scalar("Perf/motion_gather_time", motion_gather_time, it)
      writer.add_scalar("Perf/motion_gather_call_count", motion_gather_call_count, it)

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Run learning and advance adaptive sampling windows by PPO iteration."""
    _bootstrap_debug(
      "learn enter "
      f"num_learning_iterations={num_learning_iterations} "
      f"init_at_random_ep_len={init_at_random_ep_len} "
      f"is_distributed={self.is_distributed} "
      f"rank={getattr(self, 'gpu_global_rank', 'unknown')}"
    )
    if init_at_random_ep_len:
      _bootstrap_debug("before init random episode length")
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
      _bootstrap_debug("after init random episode length")

    _bootstrap_debug("before learn get_observations")
    obs = self.env.get_observations().to(self.device)
    _bootstrap_debug("after learn get_observations")
    _bootstrap_debug("before alg.train_mode")
    self.alg.train_mode()
    _bootstrap_debug("after alg.train_mode")

    if self.is_distributed:
      _bootstrap_debug(f"before broadcast_parameters rank={self.gpu_global_rank}")
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()
      _bootstrap_debug(f"after broadcast_parameters rank={self.gpu_global_rank}")

    _bootstrap_debug("before logger.init_logging_writer")
    self.logger.init_logging_writer()
    _bootstrap_debug("after logger.init_logging_writer")
    self._upload_launch_script_artifact_once()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      _bootstrap_debug(f"iteration {it}: start")
      self._step_motion_tracking_curriculum(it)
      self._begin_adaptive_sampling_iteration(it)
      self._write_large_dataset_snapshot(it)
      start = time.time()
      with torch.inference_mode():
        num_steps_per_env = int(self.cfg["num_steps_per_env"])
        _bootstrap_debug(f"iteration {it}: before rollout steps={num_steps_per_env}")
        for step_idx in range(num_steps_per_env):
          if step_idx == 0 or step_idx == num_steps_per_env - 1:
            _bootstrap_debug(f"iteration {it}: before alg.act step={step_idx}")
          actions = self.alg.act(obs)
          if step_idx == 0 or step_idx == num_steps_per_env - 1:
            _bootstrap_debug(f"iteration {it}: before env.step step={step_idx}")
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if step_idx == 0 or step_idx == num_steps_per_env - 1:
            _bootstrap_debug(f"iteration {it}: after env.step step={step_idx}")
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
          if step_idx == 0 or step_idx == num_steps_per_env - 1:
            _bootstrap_debug(f"iteration {it}: after process_env_step step={step_idx}")

        stop = time.time()
        collect_time = stop - start
        start = stop

        _bootstrap_debug(f"iteration {it}: before compute_returns")
        self.alg.compute_returns(obs)
        _bootstrap_debug(f"iteration {it}: after compute_returns")

      _bootstrap_debug(f"iteration {it}: before alg.update")
      loss_dict = self.alg.update()
      _bootstrap_debug(f"iteration {it}: after alg.update")

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      _bootstrap_debug(f"iteration {it}: before logger.log")
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=(self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None),
      )
      _bootstrap_debug(f"iteration {it}: after logger.log")
      self._log_large_dataset_timing(
        it=it, collect_time=collect_time, learn_time=learn_time
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        _bootstrap_debug(f"iteration {it}: before save")
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        _bootstrap_debug(f"iteration {it}: after save")

    if self.logger.writer is not None:
      _bootstrap_debug("before final save")
      self.save(os.path.join(self.logger.log_dir, "model_final.pt"))
      self.logger.stop_logging_writer()
      _bootstrap_debug("after final save")
    _bootstrap_debug("learn done")

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    del verbose
    checkpoint_name = getattr(
      self, "_export_checkpoint_name", f"model_{self.current_learning_iteration}.pt"
    )
    export_sim2real_policy_onnx(
      policy=self.alg.get_policy(),
      env=self.env,
      path=Path(path) / filename,
      run_name=self._run_name(),
      iteration=int(self.current_learning_iteration),
      checkpoint_name=str(checkpoint_name),
      metadata=self._deploy_metadata(),
    )

  def _run_name(self) -> str:
    return (
      wandb.run.name
      if self.logger.logger_type in {"wandb", "WandbLogWriter"} and wandb.run
      else "local"
    )

  def _env_state(self) -> dict[str, int]:
    return {"common_step_counter": int(self.env.unwrapped.common_step_counter)}

  def _deploy_metadata(self) -> dict:
    metadata = {}
    try:
      metadata = get_base_metadata(self.env.unwrapped, self._run_name())
    except Exception:
      metadata = {}
    try:
      motion_term = cast(
        MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )
      metadata.update(
        {
          "anchor_body_name": motion_term.cfg.anchor_body_name,
          "body_names": list(motion_term.cfg.body_names),
        }
      )
    except Exception:
      pass
    return metadata

  def _checkpoint_payload(self, infos=None) -> dict:
    env_state = self._env_state()
    infos = {**(infos or {}), "env_state": env_state}
    rsl_rl_state = self.alg.save()
    payload = {
      **rsl_rl_state,
      "policy": rsl_rl_state.get("actor_state_dict", {}),
      "env": env_state,
      "rsl_rl": rsl_rl_state,
      "iter": int(self.current_learning_iteration),
      "infos": infos,
    }
    if self.logger.logger_type == "wandb" and wandb.run:
      payload["wandb"] = {"name": wandb.run.name, "id": wandb.run.id}
    return payload

  def _write_deploy_metadata(self, policy_dir: Path, checkpoint_path: Path) -> None:
    metadata = {
      **self._deploy_metadata(),
      "checkpoint": checkpoint_path.name,
      "iteration": int(self.current_learning_iteration),
      "run_name": self._run_name(),
    }
    (policy_dir / "deploy_metadata.json").write_text(
      json.dumps(metadata, indent=2) + "\n"
    )

  def _export_deploy_artifacts(self, checkpoint_path: str | Path) -> None:
    policy_dir = Path(checkpoint_path).parent
    filename = "policy.onnx"
    self._export_checkpoint_name = Path(checkpoint_path).name
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
    finally:
      del self._export_checkpoint_name
    self._write_deploy_metadata(policy_dir, Path(checkpoint_path))

    onnx_path = policy_dir / filename
    writer = getattr(self.logger, "writer", None)
    if self.cfg["upload_model"] and isinstance(writer, LogWriter):
      writer.save_file(str(onnx_path))
      policy_json = onnx_path.with_suffix(".json")
      if policy_json.exists():
        writer.save_file(str(policy_json))
      deploy_metadata = policy_dir / "deploy_metadata.json"
      if deploy_metadata.exists():
        writer.save_file(str(deploy_metadata))
      if self.registry_name is not None and self.logger.logger_type in {"wandb", "WandbLogWriter"}:
        wandb.run.use_artifact(self.registry_name)  # type: ignore[union-attr]
        self.registry_name = None

  def save(self, path: str, infos=None):
    torch.save(self._checkpoint_payload(infos), path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)
    try:
      self._export_deploy_artifacts(path)
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    rsl_rl_state = dict(loaded_dict.get("rsl_rl", loaded_dict))
    if "policy" in loaded_dict and "actor_state_dict" not in rsl_rl_state:
      rsl_rl_state["actor_state_dict"] = loaded_dict["policy"]

    if "model_state_dict" in rsl_rl_state:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = rsl_rl_state.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}
      for key, value in model_state_dict.items():
        if key.startswith("actor."):
          actor_state_dict[key.replace("actor.", "mlp.")] = value
        elif key.startswith("actor_obs_normalizer."):
          actor_state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.")] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        if key.startswith("critic."):
          critic_state_dict[key.replace("critic.", "mlp.")] = value
        elif key.startswith("critic_obs_normalizer."):
          critic_state_dict[key.replace("critic_obs_normalizer.", "obs_normalizer.")] = value
      rsl_rl_state["actor_state_dict"] = actor_state_dict
      rsl_rl_state["critic_state_dict"] = critic_state_dict

    actor_sd = rsl_rl_state.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(rsl_rl_state, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = int(loaded_dict.get("iter", rsl_rl_state.get("iter", 0)))

    infos = loaded_dict.get("infos") or rsl_rl_state.get("infos") or {}
    env_state = loaded_dict.get("env")
    if not env_state and isinstance(infos, dict):
      env_state = infos.get("env_state")
    if env_state and "common_step_counter" in env_state:
      self.env.unwrapped.common_step_counter = env_state["common_step_counter"]
    return infos
