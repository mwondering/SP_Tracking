import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, cast

import mujoco
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


def _finite_mask(tensor: torch.Tensor) -> torch.Tensor:
  if tensor.is_floating_point() or tensor.is_complex():
    return torch.isfinite(tensor)
  return torch.ones_like(tensor, dtype=torch.bool)


def _nonfinite_env_ids(tensor: torch.Tensor) -> list[int]:
  if tensor.numel() == 0:
    return []
  mask = ~_finite_mask(tensor)
  if mask.ndim == 0:
    return [0] if bool(mask.item()) else []
  per_env = mask.reshape(mask.shape[0], -1).any(dim=1)
  return torch.where(per_env)[0].detach().cpu().tolist()


def _as_torch_tensor(value: object) -> torch.Tensor | None:
  """Return the tensor behind mjlab's TorchArray bridge, when available."""
  if isinstance(value, torch.Tensor):
    return value
  tensor = getattr(value, "_tensor", None)
  return tensor if isinstance(tensor, torch.Tensor) else None


_SIM_STATE_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "qacc_warmstart",
  "ctrl",
  "act",
  "act_dot",
  "qfrc_applied",
  "xfrc_applied",
  "actuator_force",
  "qfrc_actuator",
  "qfrc_constraint",
  "qacc_smooth",
  "cvel",
  "cacc",
  "cfrc_ext",
  "sensordata",
)

_SIM_FAST_CHECK_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "qacc_warmstart",
  "ctrl",
  "sensordata",
)

_SIM_INPUT_CHECK_FIELDS = (
  "ctrl",
  "qfrc_applied",
  "xfrc_applied",
)


def _get_tensor_attr(container: object, name: str) -> torch.Tensor | None:
  try:
    return _as_torch_tensor(getattr(container, name))
  except Exception:
    # Diagnostics must never replace the original simulation failure just because
    # an optional manager property is unavailable in a particular environment.
    return None


def _nonfinite_sim_env_ids(
  env: object, field_names: Iterable[str] = _SIM_FAST_CHECK_FIELDS
) -> list[int]:
  unwrapped = getattr(env, "unwrapped", env)
  sim = getattr(unwrapped, "sim", None)
  data = getattr(sim, "data", None)
  if data is None:
    return []

  per_env: torch.Tensor | None = None
  for field_name in field_names:
    tensor = _get_tensor_attr(data, field_name)
    if tensor is None or tensor.numel() == 0 or tensor.ndim == 0:
      continue
    field_mask = (~_finite_mask(tensor)).reshape(tensor.shape[0], -1).any(dim=1)
    per_env = field_mask if per_env is None else per_env | field_mask
  if per_env is None:
    return []
  return torch.where(per_env)[0].detach().cpu().tolist()


def _mj_name(model: object, object_type: mujoco.mjtObj, index: int) -> str:
  try:
    name = mujoco.mj_id2name(model, object_type, index)
  except (TypeError, ValueError):
    return ""
  return str(name) if name else ""


def _joint_name_for_address(model: object, address: int, *, qpos: bool) -> str:
  adr_name = "jnt_qposadr" if qpos else "jnt_dofadr"
  count_name = "nq" if qpos else "nv"
  try:
    addresses = getattr(model, adr_name)
    njnt = int(getattr(model, "njnt"))
    total = int(getattr(model, count_name))
  except (AttributeError, TypeError, ValueError):
    return ""
  for joint_id in range(njnt):
    start = int(addresses[joint_id])
    end = int(addresses[joint_id + 1]) if joint_id + 1 < njnt else total
    if start <= address < end:
      name = _mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
      component = address - start
      return f"joint={name or joint_id} component={component}"
  return ""


def _describe_sim_index(model: object, field_name: str, index: tuple[int, ...]) -> str:
  if not index:
    return ""
  address = index[0]
  if field_name == "qpos":
    return _joint_name_for_address(model, address, qpos=True)
  if field_name in {
    "qvel",
    "qacc",
    "qacc_warmstart",
    "qfrc_applied",
    "qfrc_actuator",
    "qfrc_constraint",
    "qacc_smooth",
  }:
    return _joint_name_for_address(model, address, qpos=False)
  if field_name in {"ctrl", "act", "act_dot", "actuator_force"}:
    name = _mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, address)
    return f"actuator={name or address}"
  if field_name in {"xfrc_applied", "cvel", "cacc", "cfrc_ext"}:
    name = _mj_name(model, mujoco.mjtObj.mjOBJ_BODY, address)
    component = index[1] if len(index) > 1 else None
    suffix = f" component={component}" if component is not None else ""
    return f"body={name or address}{suffix}"
  if field_name == "sensordata":
    try:
      sensor_adr = getattr(model, "sensor_adr")
      sensor_dim = getattr(model, "sensor_dim")
      nsensor = int(getattr(model, "nsensor"))
      for sensor_id in range(nsensor):
        start = int(sensor_adr[sensor_id])
        if start <= address < start + int(sensor_dim[sensor_id]):
          name = _mj_name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
          return f"sensor={name or sensor_id} component={address - start}"
    except (AttributeError, TypeError, ValueError):
      pass
  return ""


def _format_tensor_nonfinite_entries(
  label: str,
  tensor: torch.Tensor,
  *,
  model: object | None = None,
  field_name: str = "",
  max_entries: int = 8,
) -> list[str]:
  bad = torch.nonzero(~_finite_mask(tensor), as_tuple=False)
  if bad.numel() == 0:
    return []
  entries: list[str] = []
  for raw_index in bad[:max_entries].detach().cpu().tolist():
    env_id = int(raw_index[0]) if tensor.ndim > 0 else 0
    item_index = tuple(int(value) for value in raw_index[1:])
    value = tensor[tuple(raw_index)].detach().cpu().item()
    detail = (
      _describe_sim_index(model, field_name, item_index)
      if model is not None
      else ""
    )
    detail_suffix = f" {detail}" if detail else ""
    entries.append(
      f"env={env_id} index={item_index} value={value}{detail_suffix}"
    )
  omitted = int(bad.shape[0]) - len(entries)
  omitted_suffix = f" (+{omitted} more)" if omitted > 0 else ""
  return [f"- {label}: " + "; ".join(entries) + omitted_suffix]


def _format_simulation_counts(env: object, env_ids: list[int]) -> list[str]:
  unwrapped = getattr(env, "unwrapped", env)
  sim = getattr(unwrapped, "sim", None)
  data = getattr(sim, "data", None)
  if data is None:
    return []
  parts: list[str] = []
  for name in ("nefc", "nacon", "ncollision", "solver_niter"):
    tensor = _get_tensor_attr(data, name)
    if tensor is None or tensor.numel() == 0:
      continue
    flat = tensor.reshape(tensor.shape[0], -1)
    valid_ids = [env_id for env_id in env_ids if env_id < flat.shape[0]]
    values = {
      env_id: flat[env_id].detach().cpu().tolist() for env_id in valid_ids[:10]
    }
    maximum = flat.max().detach().cpu().item()
    parts.append(f"{name}(bad_envs)={values} {name}_global_max={maximum}")
  return ["- solver/contact counters: " + " | ".join(parts)] if parts else []


def _format_constraint_type_counts(env: object, env_ids: list[int]) -> list[str]:
  unwrapped = getattr(env, "unwrapped", env)
  sim = getattr(unwrapped, "sim", None)
  data = getattr(sim, "data", None)
  efc = getattr(data, "efc", None)
  constraint_types = _get_tensor_attr(efc, "type")
  nefc = _get_tensor_attr(data, "nefc")
  if constraint_types is None or nefc is None:
    return []

  parts: list[str] = []
  for env_id in env_ids[:5]:
    if env_id >= constraint_types.shape[0] or env_id >= nefc.shape[0]:
      continue
    count = min(int(nefc[env_id].detach().cpu().item()), constraint_types.shape[1])
    if count <= 0:
      continue
    values, counts = torch.unique(
      constraint_types[env_id, :count], return_counts=True
    )
    type_counts: dict[str, int] = {}
    for raw_type, raw_count in zip(
      values.detach().cpu().tolist(), counts.detach().cpu().tolist(), strict=True
    ):
      try:
        type_name = mujoco.mjtConstraint(int(raw_type)).name
      except ValueError:
        type_name = str(int(raw_type))
      type_counts[type_name] = int(raw_count)
    parts.append(f"env={env_id} nefc={int(nefc[env_id].item())} types={type_counts}")
  return ["- active constraint composition: " + " | ".join(parts)] if parts else []


def _format_contact_geom_pairs(env: object, env_ids: list[int]) -> list[str]:
  unwrapped = getattr(env, "unwrapped", env)
  sim = getattr(unwrapped, "sim", None)
  data = getattr(sim, "data", None)
  model = getattr(sim, "mj_model", None)
  contact = getattr(data, "contact", None)
  world_ids = _get_tensor_attr(contact, "worldid")
  geom_pairs = _get_tensor_attr(contact, "geom")
  if world_ids is None or geom_pairs is None:
    return []

  flat_world_ids = world_ids.reshape(-1)
  flat_geom_pairs = geom_pairs.reshape(-1, 2)
  parts: list[str] = []
  for env_id in env_ids[:5]:
    pairs = flat_geom_pairs[flat_world_ids == env_id]
    pairs = pairs[(pairs >= 0).all(dim=1)]
    if pairs.numel() == 0:
      continue
    unique_pairs, counts = torch.unique(pairs, dim=0, return_counts=True)
    order = torch.argsort(counts, descending=True)[:12]
    descriptions: list[str] = []
    for pair, count in zip(
      unique_pairs[order].detach().cpu().tolist(),
      counts[order].detach().cpu().tolist(),
      strict=True,
    ):
      geom_a, geom_b = int(pair[0]), int(pair[1])
      name_a = _mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
      name_b = _mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
      descriptions.append(
        f"({name_a or geom_a}, {name_b or geom_b})x{int(count)}"
      )
    parts.append(f"env={env_id} " + ", ".join(descriptions))
  return ["- active contact geom pairs: " + " | ".join(parts)] if parts else []


def _format_nonfinite_internal_state_diagnostics(
  env: object, preferred_env_ids: Iterable[int] = ()
) -> str:
  """Describe exact non-finite indices in simulator, action, and motion state."""
  lines: list[str] = ["Internal non-finite source diagnostics:"]
  unwrapped = getattr(env, "unwrapped", env)
  sim = getattr(unwrapped, "sim", None)
  data = getattr(sim, "data", None)
  model = getattr(sim, "mj_model", None)
  all_bad_env_ids: set[int] = set(int(value) for value in preferred_env_ids)

  if data is not None:
    for field_name in _SIM_STATE_FIELDS:
      tensor = _get_tensor_attr(data, field_name)
      if tensor is None:
        continue
      field_env_ids = _nonfinite_env_ids(tensor)
      all_bad_env_ids.update(field_env_ids)
      if field_env_ids:
        lines.extend(
          _format_tensor_nonfinite_entries(
            f"sim.data.{field_name}",
            tensor,
            model=model,
            field_name=field_name,
          )
        )

  action_manager = getattr(unwrapped, "action_manager", None)
  for attr_name in ("action", "prev_action"):
    tensor = _get_tensor_attr(action_manager, attr_name)
    if tensor is not None:
      lines.extend(_format_tensor_nonfinite_entries(f"action_manager.{attr_name}", tensor))
  get_action_term = getattr(action_manager, "get_term", None)
  if callable(get_action_term):
    for term_name in getattr(action_manager, "active_terms", ()):
      try:
        term = get_action_term(term_name)
      except Exception:
        continue
      for attr_name in (
        "_raw_actions",
        "_processed_actions",
        "applied_action",
        "joint_offset",
        "alpha",
        "boot_target",
      ):
        tensor = _get_tensor_attr(term, attr_name)
        if tensor is not None:
          lines.extend(
            _format_tensor_nonfinite_entries(
              f"action[{term_name}].{attr_name}", tensor
            )
          )

  command_manager = getattr(unwrapped, "command_manager", None)
  get_command_term = getattr(command_manager, "get_term", None)
  if callable(get_command_term):
    try:
      command = get_command_term("motion")
    except Exception:
      command = None
    if command is not None:
      for attr_name in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ):
        tensor = _get_tensor_attr(command, attr_name)
        if tensor is not None:
          lines.extend(
            _format_tensor_nonfinite_entries(
              f"motion_reference.{attr_name}", tensor
            )
          )

  bad_env_ids = sorted(all_bad_env_ids)
  if bad_env_ids:
    context = _motion_context(env, bad_env_ids)
    if context:
      lines.append(f"- failing motion context: envs={bad_env_ids[:10]} {context}")
    lines.extend(_format_simulation_counts(env, bad_env_ids))
    lines.extend(_format_constraint_type_counts(env, bad_env_ids))
    lines.extend(_format_contact_geom_pairs(env, bad_env_ids))
  return "\n".join(lines) if len(lines) > 1 else ""


def _safe_internal_state_diagnostics(
  env: object, preferred_env_ids: Iterable[int] = ()
) -> str:
  try:
    return _format_nonfinite_internal_state_diagnostics(env, preferred_env_ids)
  except Exception as exc:
    return f"Internal diagnostics failed without replacing the original error: {exc!r}"


class _FirstNonfiniteSimulationTracer:
  """Stop at the first physics substep whose MuJoCo state becomes non-finite."""

  def __init__(self, env: object):
    self.env = env
    unwrapped = getattr(env, "unwrapped", env)
    self.sim = getattr(unwrapped, "sim")
    self._original_step = self.sim.step
    self.iteration = -1
    self.rollout_step = -1
    self.physics_substep = 0
    self.global_physics_step = 0
    self.sim.step = self.step

    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    print(
      "[NONFINITE_TRACE] enabled for physics substeps "
      f"rank={rank} local_rank={local_rank}; "
      "set task.debug_nonfinite_state=false to disable",
      flush=True,
    )

    initial_bad_env_ids = _nonfinite_sim_env_ids(env)
    if initial_bad_env_ids:
      raise FloatingPointError(
        self._message("runner initialization", initial_bad_env_ids)
      )

  def begin_env_step(self, iteration: int, rollout_step: int) -> None:
    self.iteration = int(iteration)
    self.rollout_step = int(rollout_step)
    self.physics_substep = 0

  def _message(self, phase: str, env_ids: list[int]) -> str:
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    header = (
      "[FIRST_NONFINITE_STATE] "
      f"rank={rank} local_rank={local_rank} phase={phase} "
      f"iteration={self.iteration} rollout_step={self.rollout_step} "
      f"physics_substep={self.physics_substep} "
      f"global_physics_step={self.global_physics_step} envs={env_ids[:10]}"
    )
    details = _safe_internal_state_diagnostics(self.env, env_ids)
    return f"{header}\n{details}" if details else header

  def step(self) -> None:
    self.physics_substep += 1
    self.global_physics_step += 1
    bad_input_env_ids = _nonfinite_sim_env_ids(
      self.env, _SIM_INPUT_CHECK_FIELDS
    )
    if bad_input_env_ids:
      message = self._message("before sim.step", bad_input_env_ids)
      print(message, flush=True)
      raise FloatingPointError(message)

    self._original_step()
    bad_env_ids = _nonfinite_sim_env_ids(self.env)
    if not bad_env_ids:
      return
    message = self._message("after sim.step", bad_env_ids)
    print(message, flush=True)
    raise FloatingPointError(message)


def _motion_context(env: object, env_ids: Iterable[int]) -> str:
  unwrapped = getattr(env, "unwrapped", env)
  command_manager = getattr(unwrapped, "command_manager", None)
  get_term = getattr(command_manager, "get_term", None)
  if not callable(get_term):
    return ""
  try:
    command = get_term("motion")
  except Exception:
    return ""

  motion_idx = getattr(command, "motion_idx", None)
  time_steps = getattr(command, "time_steps", None)
  if not isinstance(motion_idx, torch.Tensor) or not isinstance(time_steps, torch.Tensor):
    return ""

  env_id_list = [env_id for env_id in env_ids if env_id < int(motion_idx.numel())]
  if not env_id_list:
    return ""
  env_id_tensor = torch.as_tensor(
    env_id_list, dtype=torch.long, device=motion_idx.device
  )
  motion_ids = motion_idx[env_id_tensor].detach().cpu().tolist()
  step_values = time_steps[env_id_tensor].detach().cpu().tolist()
  parts = [f"motion_ids={motion_ids}", f"time_steps={step_values}"]

  motion_store = getattr(command, "motion_store", None)
  motion_files = getattr(motion_store, "motion_files", None)
  if isinstance(motion_files, list):
    file_examples = []
    for motion_id in motion_ids[:5]:
      if isinstance(motion_id, int) and 0 <= motion_id < len(motion_files):
        file_examples.append(str(motion_files[motion_id]))
    if file_examples:
      parts.append(f"motion_files={file_examples}")
  return " ".join(parts)


def _diagnose_concatenated_obs_group(
  observation_manager: object,
  group_name: str,
  tensor: torch.Tensor,
) -> list[tuple[str, list[int]]]:
  active_terms = getattr(observation_manager, "active_terms", {})
  group_obs_term_dim = getattr(observation_manager, "group_obs_term_dim", {})
  group_obs_concatenate = getattr(observation_manager, "group_obs_concatenate", {})
  if not isinstance(active_terms, dict) or not isinstance(group_obs_term_dim, dict):
    return []
  if not group_obs_concatenate.get(group_name, False):
    return []
  term_names = active_terms.get(group_name, [])
  term_dims = group_obs_term_dim.get(group_name, [])
  if not term_names or not term_dims:
    return []

  flat = tensor.reshape(tensor.shape[0], -1)
  cursor = 0
  results: list[tuple[str, list[int]]] = []
  for term_name, term_dim in zip(term_names, term_dims, strict=False):
    term_width = int(math.prod(term_dim))
    term = flat[:, cursor : cursor + term_width]
    env_ids = _nonfinite_env_ids(term)
    if env_ids:
      results.append((str(term_name), env_ids))
    cursor += term_width
  return results


def _format_nonfinite_env_output_diagnostics(
  env: object,
  obs: object,
  rewards: torch.Tensor,
  dones: torch.Tensor,
) -> str:
  lines: list[str] = ["Non-finite environment output diagnostics:"]
  unwrapped = getattr(env, "unwrapped", env)
  observation_manager = getattr(unwrapped, "observation_manager", None)

  obs_items = getattr(obs, "items", None)
  if callable(obs_items):
    for group_name, tensor in obs_items():
      if not isinstance(tensor, torch.Tensor):
        continue
      group_env_ids = _nonfinite_env_ids(tensor)
      if not group_env_ids:
        continue
      term_hits = (
        _diagnose_concatenated_obs_group(observation_manager, str(group_name), tensor)
        if observation_manager is not None
        else []
      )
      if term_hits:
        for term_name, env_ids in term_hits[:12]:
          motion_context = _motion_context(env, env_ids)
          suffix = f" {motion_context}" if motion_context else ""
          lines.append(f"- obs[{group_name}]/{term_name}: envs={env_ids[:10]}{suffix}")
      else:
        motion_context = _motion_context(env, group_env_ids)
        suffix = f" {motion_context}" if motion_context else ""
        lines.append(f"- obs[{group_name}]: envs={group_env_ids[:10]}{suffix}")

  reward_env_ids = _nonfinite_env_ids(rewards)
  if reward_env_ids:
    lines.append(f"- rewards: envs={reward_env_ids[:10]}")
  done_env_ids = _nonfinite_env_ids(dones)
  if done_env_ids:
    lines.append(f"- dones: envs={done_env_ids[:10]}")

  if len(lines) == 1:
    return ""
  return "\n".join(lines)


def _format_nonfinite_action_diagnostics(env: object, actions: torch.Tensor) -> str:
  env_ids = _nonfinite_env_ids(actions)
  if not env_ids:
    return ""
  motion_context = _motion_context(env, env_ids)
  suffix = f" {motion_context}" if motion_context else ""
  return f"Policy action contains NaN/Inf: envs={env_ids[:10]}{suffix}"


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
    debug_nonfinite_state: bool = False,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name
    self.launch_script_artifact_path = launch_script_artifact_path
    self._launch_script_artifact_uploaded = False
    self._nonfinite_tracer = (
      _FirstNonfiniteSimulationTracer(self.env) if debug_nonfinite_state else None
    )

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

  def _record_policy_action_mean(self) -> None:
    """Expose the sampled policy's mean to action-history observations.

    RSL-RL keeps the distribution mean on the actor after ``act``.  The SP
    action term opts into this hook through ``prev_action_obs: mean``; other
    action terms simply do not implement ``record_policy_mean``.
    """
    actor = getattr(self.alg, "actor", None)
    mean = getattr(actor, "output_mean", None)
    if not isinstance(mean, torch.Tensor):
      return
    action_manager = getattr(self.env.unwrapped, "action_manager", None)
    get_term = getattr(action_manager, "get_term", None)
    if not callable(get_term):
      return
    try:
      action_term = get_term("joint_pos")
    except KeyError:
      return
    record_mean = getattr(action_term, "record_policy_mean", None)
    if callable(record_mean):
      record_mean(mean)

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
          self._record_policy_action_mean()
          if self.cfg.get("check_for_nan", True):
            action_diagnostics = _format_nonfinite_action_diagnostics(
              self.env, actions
            )
            if action_diagnostics:
              raise ValueError(action_diagnostics)
          if step_idx == 0 or step_idx == num_steps_per_env - 1:
            _bootstrap_debug(f"iteration {it}: before env.step step={step_idx}")
          if self._nonfinite_tracer is not None:
            self._nonfinite_tracer.begin_env_step(it, step_idx)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if step_idx == 0 or step_idx == num_steps_per_env - 1:
            _bootstrap_debug(f"iteration {it}: after env.step step={step_idx}")
          if self.cfg.get("check_for_nan", True):
            try:
              check_nan(obs, rewards, dones)
            except ValueError as exc:
              diagnostics = _format_nonfinite_env_output_diagnostics(
                self.env, obs, rewards, dones
              )
              failing_env_ids: set[int] = set()
              obs_items = getattr(obs, "items", None)
              if callable(obs_items):
                for _, tensor in obs_items():
                  if isinstance(tensor, torch.Tensor):
                    failing_env_ids.update(_nonfinite_env_ids(tensor))
              failing_env_ids.update(_nonfinite_env_ids(rewards))
              internal_diagnostics = _safe_internal_state_diagnostics(
                self.env, sorted(failing_env_ids)
              )
              combined = "\n".join(
                value for value in (diagnostics, internal_diagnostics) if value
              )
              if combined:
                raise ValueError(f"{exc}\n{combined}") from exc
              raise
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
