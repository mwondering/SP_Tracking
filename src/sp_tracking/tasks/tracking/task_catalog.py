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
  "tracking_bfm_student_actor_bfm_critic",
  "tracking_bfm_teacher_actor_bfm_critic",
  "tracking_bfm_wbteleop_actor_bfm_critic",
  "tracking_bfm_wbteleop_actor_heft_critic",
]


@dataclass(frozen=True)
class TaskSpec:
  name: TaskName
  task_id: str
  hydra_overrides: tuple[str, ...]


TASK_SPECS = (
  TaskSpec(
    "tracking_bfm",
    "SPTracking-G1-BFM-BFMActor-BFMCritic",
    (),
  ),
  TaskSpec(
    "tracking_bfm_sp",
    "SPTracking-G1-HEFT-TeacherActor-HEFTCritic",
    ("task=tracking_bfm_sp",),
  ),
  TaskSpec(
    "tracking_bfm_sp_ablation_bfm_actor",
    "SPTracking-G1-BFM-BFMActor-HEFTCritic",
    ("task=tracking_bfm_sp_ablation_bfm_actor",),
  ),
  TaskSpec(
    "tracking_bfm_sp_ablation_student_actor",
    "SPTracking-G1-BFM-StudentActor-HEFTCritic",
    ("task=tracking_bfm_sp_ablation_student_actor",),
  ),
  TaskSpec(
    "tracking_bfm_sp_ablation_teacher_actor",
    "SPTracking-G1-BFM-TeacherActor-HEFTCritic",
    ("task=tracking_bfm_sp_ablation_teacher_actor",),
  ),
  TaskSpec(
    "tracking_bfm_student_actor_bfm_critic",
    "SPTracking-G1-BFM-StudentActor-BFMCritic",
    ("task=tracking_bfm_student_actor_bfm_critic",),
  ),
  TaskSpec(
    "tracking_bfm_teacher_actor_bfm_critic",
    "SPTracking-G1-BFM-TeacherActor-BFMCritic",
    ("task=tracking_bfm_teacher_actor_bfm_critic",),
  ),
  TaskSpec(
    "tracking_bfm_wbteleop_actor_bfm_critic",
    "SPTracking-G1-BFM-WBTeleopActor-BFMCritic",
    ("task=tracking_bfm_wbteleop_actor_bfm_critic",),
  ),
  TaskSpec(
    "tracking_bfm_wbteleop_actor_heft_critic",
    "SPTracking-G1-BFM-WBTeleopActor-HEFTCritic",
    ("task=tracking_bfm_wbteleop_actor_heft_critic",),
  ),
)

TASK_BY_NAME = {spec.name: spec for spec in TASK_SPECS}
