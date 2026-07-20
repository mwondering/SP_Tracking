"""Single source of truth for supported tracking tasks."""

from __future__ import annotations

from dataclasses import dataclass


TaskId = str


@dataclass(frozen=True)
class TaskSpec:
  # ``config_name`` is a private Hydra implementation key.  ``task_id`` is
  # the sole user-facing and registry-facing task identifier.
  config_name: str
  task_id: TaskId
  hydra_overrides: tuple[str, ...]
  is_experiment: bool = False


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
  TaskSpec(
    "tracking_bfm_heft_reward",
    "SPTracking-G1-BFM-BFMActor-BFMCritic-HEFTReward",
    ("task=tracking_bfm_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_sp_ablation_bfm_actor_heft_reward",
    "SPTracking-G1-BFM-BFMActor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_sp_ablation_bfm_actor_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_sp_ablation_student_actor_heft_reward",
    "SPTracking-G1-BFM-StudentActor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_sp_ablation_student_actor_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_sp_ablation_teacher_actor_heft_reward",
    "SPTracking-G1-BFM-TeacherActor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_sp_ablation_teacher_actor_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_student_actor_bfm_critic_heft_reward",
    "SPTracking-G1-BFM-StudentActor-BFMCritic-HEFTReward",
    ("task=tracking_bfm_student_actor_bfm_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_teacher_actor_bfm_critic_heft_reward",
    "SPTracking-G1-BFM-TeacherActor-BFMCritic-HEFTReward",
    ("task=tracking_bfm_teacher_actor_bfm_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_wbteleop_actor_bfm_critic_heft_reward",
    "SPTracking-G1-BFM-WBTeleopActor-BFMCritic-HEFTReward",
    ("task=tracking_bfm_wbteleop_actor_bfm_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_wbteleop_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-WBTeleopActor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_wbteleop_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv1_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV1Actor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv1_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv2_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV2Actor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv2_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv3_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV3Actor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv3_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv4_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV4Actor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv4_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv5_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV5Actor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv5_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv5_1_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV5-1Actor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv5_1_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv5_1_moe_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV5-1MoEActor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv5_1_moe_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv5_1_actor_heft_moe_critic_heft_reward",
    "SPTracking-G1-BFM-SPV5-1Actor-HEFTMoECritic-HEFTReward",
    ("task=tracking_bfm_spv5_1_actor_heft_moe_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv6_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV6Actor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv6_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv6_1_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV6-1Actor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv6_1_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "tracking_bfm_spv6_0_actor_heft_critic_heft_reward",
    "SPTracking-G1-BFM-SPV6-0Actor-HEFTCritic-HEFTReward",
    ("task=tracking_bfm_spv6_0_actor_heft_critic_heft_reward",),
  ),
  TaskSpec(
    "test_policy_gradients",
    "SPTracking-G1-TestPolicyGradients",
    ("task=test_policy_gradients",),
    is_experiment=True,
  ),
)

TASK_BY_CONFIG_NAME = {spec.config_name: spec for spec in TASK_SPECS}
TASK_BY_ID = {spec.task_id: spec for spec in TASK_SPECS}
