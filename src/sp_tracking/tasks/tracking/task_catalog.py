"""Single source of truth for supported tracking tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskName = Literal[
  "tracking_bfm",
  "tracking_bfm_sp",
  "tracking_bfm_sp_ablation_bfm_actor",
  "tracking_bfm_sp_ablation_student_actor",
  "tracking_bfm_sp_ablation_teacher_actor",
]


@dataclass(frozen=True)
class TaskSpec:
  name: TaskName
  task_id: str
  hydra_overrides: tuple[str, ...]


TASK_SPECS = (
  TaskSpec("tracking_bfm", "SPTracking-G1-BFM", ()),
  TaskSpec("tracking_bfm_sp", "SPTracking-G1-BFM-SP", ("task=tracking_bfm_sp",)),
  TaskSpec(
    "tracking_bfm_sp_ablation_bfm_actor",
    "SPTracking-G1-BFM-SP-Ablation-BFMActor",
    ("task=tracking_bfm_sp_ablation_bfm_actor",),
  ),
  TaskSpec(
    "tracking_bfm_sp_ablation_student_actor",
    "SPTracking-G1-BFM-SP-Ablation-StudentActor",
    ("task=tracking_bfm_sp_ablation_student_actor",),
  ),
  TaskSpec(
    "tracking_bfm_sp_ablation_teacher_actor",
    "SPTracking-G1-BFM-SP-Ablation-TeacherActor",
    ("task=tracking_bfm_sp_ablation_teacher_actor",),
  ),
)

TASK_BY_NAME = {spec.name: spec for spec in TASK_SPECS}

