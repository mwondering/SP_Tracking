from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from sp_tracking.tasks.tracking.mdp.multi_command_largedataset import (
  MotionCommandCfg as LargeDatasetMotionCommandCfg,
)
from sp_tracking.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from sp_tracking.tasks.tracking.rl import SpTrackingOnPolicyRunner


def test_tracking_entrypoint_registers_default_tasks() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  tasks = list_tasks()

  assert registry.TRACKING_BFM_TASK_ID in tasks
  assert registry.TRACKING_BFM_SP_TASK_ID in tasks
  assert registry.TRACKING_BFM_SP_ABLATION_BFM_ACTOR_TASK_ID in tasks
  assert registry.TRACKING_BFM_SP_ABLATION_STUDENT_ACTOR_TASK_ID in tasks
  assert registry.TRACKING_BFM_SP_ABLATION_TEACHER_ACTOR_TASK_ID in tasks
  assert registry.TRACKING_BFM_STUDENT_ACTOR_BFM_CRITIC_TASK_ID in tasks
  assert registry.TRACKING_BFM_TEACHER_ACTOR_BFM_CRITIC_TASK_ID in tasks
  assert registry.TRACKING_BFM_WBTELEOP_ACTOR_BFM_CRITIC_TASK_ID in tasks
  assert registry.TRACKING_BFM_WBTELEOP_ACTOR_HEFT_CRITIC_TASK_ID in tasks
  assert registry.TRACKING_BFM_SPV1_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID in tasks
  assert registry.TRACKING_BFM_SPV2_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID in tasks
  assert registry.TRACKING_BFM_SPV3_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID in tasks
  assert registry.TRACKING_BFM_SPV4_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID in tasks
  assert registry.TRACKING_BFM_SPV5_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID in tasks
  assert registry.TRACKING_BFM_SPV6_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID in tasks
  assert registry.TRACKING_BFM_SPV6_1_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID in tasks
  assert registry.TRACKING_BFM_SPV6_0_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID in tasks


def test_task_ids_describe_runtime_actor_and_critic_semantics() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  assert registry.TRACKING_BFM_TASK_ID == (
    "SPTracking-G1-BFM-BFMActor-BFMCritic"
  )
  assert registry.TRACKING_BFM_SP_TASK_ID == (
    "SPTracking-G1-HEFT-TeacherActor-HEFTCritic"
  )
  assert registry.TRACKING_BFM_SP_ABLATION_BFM_ACTOR_TASK_ID == (
    "SPTracking-G1-BFM-BFMActor-HEFTCritic"
  )
  assert registry.TRACKING_BFM_SP_ABLATION_STUDENT_ACTOR_TASK_ID == (
    "SPTracking-G1-BFM-StudentActor-HEFTCritic"
  )
  assert registry.TRACKING_BFM_SP_ABLATION_TEACHER_ACTOR_TASK_ID == (
    "SPTracking-G1-BFM-TeacherActor-HEFTCritic"
  )
  assert registry.TRACKING_BFM_STUDENT_ACTOR_BFM_CRITIC_TASK_ID == (
    "SPTracking-G1-BFM-StudentActor-BFMCritic"
  )
  assert registry.TRACKING_BFM_TEACHER_ACTOR_BFM_CRITIC_TASK_ID == (
    "SPTracking-G1-BFM-TeacherActor-BFMCritic"
  )
  assert registry.TRACKING_BFM_WBTELEOP_ACTOR_BFM_CRITIC_TASK_ID == (
    "SPTracking-G1-BFM-WBTeleopActor-BFMCritic"
  )
  assert registry.TRACKING_BFM_WBTELEOP_ACTOR_HEFT_CRITIC_TASK_ID == (
    "SPTracking-G1-BFM-WBTeleopActor-HEFTCritic"
  )
  assert registry.TRACKING_BFM_SPV1_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID == (
    "SPTracking-G1-BFM-SPV1Actor-HEFTCritic-HEFTReward"
  )
  assert registry.TRACKING_BFM_SPV2_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID == (
    "SPTracking-G1-BFM-SPV2Actor-HEFTCritic-HEFTReward"
  )
  assert registry.TRACKING_BFM_SPV3_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID == (
    "SPTracking-G1-BFM-SPV3Actor-HEFTCritic-HEFTReward"
  )
  assert registry.TRACKING_BFM_SPV4_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID == (
    "SPTracking-G1-BFM-SPV4Actor-HEFTCritic-HEFTReward"
  )
  assert registry.TRACKING_BFM_SPV5_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID == (
    "SPTracking-G1-BFM-SPV5Actor-HEFTCritic-HEFTReward"
  )
  assert registry.TRACKING_BFM_SPV6_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID == (
    "SPTracking-G1-BFM-SPV6Actor-HEFTCritic-HEFTReward"
  )
  assert registry.TRACKING_BFM_SPV6_1_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID == (
    "SPTracking-G1-BFM-SPV6-1Actor-HEFTCritic-HEFTReward"
  )
  assert registry.TRACKING_BFM_SPV6_0_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID == (
    "SPTracking-G1-BFM-SPV6-0Actor-HEFTCritic-HEFTReward"
  )


def test_registered_default_task_loads_hydra_built_configs() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  env_cfg = load_env_cfg(registry.TRACKING_BFM_TASK_ID)
  rl_cfg = load_rl_cfg(registry.TRACKING_BFM_TASK_ID)

  assert isinstance(env_cfg.commands["motion"], MultiMotionCommandCfg)
  assert rl_cfg.experiment_name == "g1_tracking"
  assert load_runner_cls(registry.TRACKING_BFM_TASK_ID) is SpTrackingOnPolicyRunner


def test_registered_sp_task_uses_large_dataset_command() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  env_cfg = load_env_cfg(registry.TRACKING_BFM_SP_TASK_ID)

  assert isinstance(env_cfg.commands["motion"], LargeDatasetMotionCommandCfg)
  assert "motor_params_implicit" in env_cfg.events
  assert "sp_tracking_progress" in env_cfg.curriculum


def test_registered_ablation_uses_bfm_runtime_and_sp_observations() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  env_cfg = load_env_cfg(registry.TRACKING_BFM_SP_ABLATION_STUDENT_ACTOR_TASK_ID)
  rl_cfg = load_rl_cfg(registry.TRACKING_BFM_SP_ABLATION_STUDENT_ACTOR_TASK_ID)

  assert isinstance(env_cfg.commands["motion"], MultiMotionCommandCfg)
  assert env_cfg.commands["motion"].history_steps == 0
  assert env_cfg.commands["motion"].future_steps == 1
  assert "motion_global_root_pos" in env_cfg.rewards
  assert "anchor_pos" in env_cfg.terminations
  assert rl_cfg.experiment_name == "g1_tracking"
  assert rl_cfg.obs_groups == {
    "actor": ("policy",),
    "critic": ("policy", "priv"),
  }


def test_registered_student_baseline_uses_original_bfm_critic() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  env_cfg = load_env_cfg(registry.TRACKING_BFM_STUDENT_ACTOR_BFM_CRITIC_TASK_ID)
  rl_cfg = load_rl_cfg(registry.TRACKING_BFM_STUDENT_ACTOR_BFM_CRITIC_TASK_ID)

  assert tuple(env_cfg.observations) == ("actor", "critic", "policy", "priv")
  assert rl_cfg.obs_groups == {
    "actor": ("policy",),
    "critic": ("critic",),
  }
  assert rl_cfg.critic.class_name == "MLPModel"
  assert rl_cfg.critic.hidden_dims == (2048, 2048, 1024, 1024, 512, 256, 128)


def test_registered_wbteleop_comparison_uses_heft_critic() -> None:
  import sp_tracking.tasks.tracking.registry as registry

  env_cfg = load_env_cfg(
    registry.TRACKING_BFM_WBTELEOP_ACTOR_HEFT_CRITIC_TASK_ID
  )
  rl_cfg = load_rl_cfg(
    registry.TRACKING_BFM_WBTELEOP_ACTOR_HEFT_CRITIC_TASK_ID
  )

  assert tuple(env_cfg.observations) == ("actor", "policy", "priv")
  assert rl_cfg.obs_groups == {
    "actor": ("actor",),
    "critic": ("policy", "priv"),
  }
  assert rl_cfg.critic.class_name.endswith(":HeftTeacherCritic")
  assert rl_cfg.critic.hidden_dims == (1024, 512, 512)
