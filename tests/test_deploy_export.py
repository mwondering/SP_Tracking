import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sp_tracking.tasks.tracking.rl.export import (
  build_sim2real_policy_metadata,
  export_sim2real_policy_onnx,
)


class _FakePolicy(torch.nn.Module):
  input_size = 7
  input_names = ["obs"]
  output_names = ["actions"]

  def __init__(self) -> None:
    super().__init__()
    self.linear = torch.nn.Linear(7, 3)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.linear(obs)

  def as_onnx(self, verbose: bool = False):
    del verbose
    return self

  def get_dummy_inputs(self):
    return (torch.zeros(1, self.input_size),)


def test_build_sim2real_policy_metadata_uses_policy_and_action_keys() -> None:
  metadata = build_sim2real_policy_metadata(
    env=SimpleNamespace(num_actions=3),
    policy=_FakePolicy(),
    run_name="run_a",
    iteration=12,
    checkpoint_name="model_12.pt",
  )

  assert metadata["in_keys"] == ["policy"]
  assert metadata["out_keys"] == ["action"]
  assert metadata["in_shapes"] == [[[1, 7]]]
  assert metadata["num_actions"] == 3
  assert metadata["checkpoint"] == "model_12.pt"


def test_export_sim2real_policy_onnx_writes_policy_json(tmp_path: Path) -> None:
  onnx_path = tmp_path / "policy.onnx"

  export_sim2real_policy_onnx(
    policy=_FakePolicy(),
    env=SimpleNamespace(num_actions=3),
    path=onnx_path,
    run_name="local",
    iteration=5,
    checkpoint_name="model_5.pt",
  )

  metadata = json.loads((tmp_path / "policy.json").read_text())
  assert onnx_path.exists()
  assert metadata["in_keys"] == ["policy"]
  assert metadata["out_keys"] == ["action"]
