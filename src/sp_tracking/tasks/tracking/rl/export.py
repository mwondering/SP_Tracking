from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


# External deployment contract retained for compatibility with the reference
# framework.  It is intentionally not an internal SP naming choice.
REFERENCE_SIM2REAL_POLICY_FORMAT = "motion_tracking_sim2real_policy"


def _policy_input_size(policy: torch.nn.Module) -> int:
  if hasattr(policy, "input_size"):
    return int(getattr(policy, "input_size"))
  dummy_inputs = policy.get_dummy_inputs()  # type: ignore[attr-defined]
  return int(dummy_inputs[0].shape[-1])


def build_sim2real_policy_metadata(
  *,
  env: Any,
  policy: torch.nn.Module,
  run_name: str,
  iteration: int | None,
  checkpoint_name: str,
) -> dict[str, Any]:
  input_size = _policy_input_size(policy)
  return {
    "format": REFERENCE_SIM2REAL_POLICY_FORMAT,
    "run_name": run_name,
    "iteration": iteration,
    "checkpoint": checkpoint_name,
    "in_keys": ["policy"],
    "out_keys": ["action"],
    "in_shapes": [[[1, input_size]]],
    "num_actions": int(getattr(env, "num_actions")),
  }


@torch.inference_mode()
def export_sim2real_policy_onnx(
  *,
  policy: torch.nn.Module,
  env: Any,
  path: str | Path,
  run_name: str,
  iteration: int | None,
  checkpoint_name: str,
  metadata: dict[str, Any] | None = None,
) -> None:
  path = Path(path)
  if path.suffix != ".onnx":
    raise ValueError(f"Export path must end with .onnx, got {path}")
  path.parent.mkdir(parents=True, exist_ok=True)

  onnx_model = policy.as_onnx(verbose=False) if hasattr(policy, "as_onnx") else policy
  onnx_model.to("cpu")
  onnx_model.eval()
  input_size = _policy_input_size(onnx_model)
  dummy_obs = torch.zeros(1, input_size, dtype=torch.float32)
  torch.onnx.export(
    onnx_model,
    (dummy_obs,),
    str(path),
    export_params=True,
    opset_version=18,
    input_names=["policy"],
    output_names=["action"],
    dynamic_axes={},
    dynamo=False,
  )

  base_metadata = build_sim2real_policy_metadata(
    env=env,
    policy=onnx_model,
    run_name=run_name,
    iteration=iteration,
    checkpoint_name=checkpoint_name,
  )
  if metadata:
    base_metadata.update(metadata)
  path.with_suffix(".json").write_text(json.dumps(base_metadata, indent=2) + "\n")
