from __future__ import annotations

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from sp_tracking.config.build_agent import build_agent_cfg
from sp_tracking.config.build_env import build_env_cfg
from sp_tracking.tasks.tracking.rl import SpTrackingOnPolicyRunner
from sp_tracking.tasks.tracking.task_catalog import TASK_BY_NAME, TASK_SPECS


TRACKING_BFM_TASK_ID = TASK_BY_NAME["tracking_bfm"].task_id
TRACKING_BFM_SP_TASK_ID = TASK_BY_NAME["tracking_bfm_sp"].task_id
TRACKING_BFM_SP_ABLATION_BFM_ACTOR_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_sp_ablation_bfm_actor"
].task_id
TRACKING_BFM_SP_ABLATION_STUDENT_ACTOR_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_sp_ablation_student_actor"
].task_id
TRACKING_BFM_SP_ABLATION_TEACHER_ACTOR_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_sp_ablation_teacher_actor"
].task_id
TRACKING_BFM_STUDENT_ACTOR_BFM_CRITIC_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_student_actor_bfm_critic"
].task_id
TRACKING_BFM_TEACHER_ACTOR_BFM_CRITIC_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_teacher_actor_bfm_critic"
].task_id
TRACKING_BFM_WBTELEOP_ACTOR_BFM_CRITIC_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_wbteleop_actor_bfm_critic"
].task_id
TRACKING_BFM_WBTELEOP_ACTOR_HEFT_CRITIC_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_wbteleop_actor_heft_critic"
].task_id
TRACKING_BFM_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_heft_reward"
].task_id
TRACKING_BFM_SP_ABLATION_BFM_ACTOR_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_sp_ablation_bfm_actor_heft_reward"
].task_id
TRACKING_BFM_SP_ABLATION_STUDENT_ACTOR_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_sp_ablation_student_actor_heft_reward"
].task_id
TRACKING_BFM_SP_ABLATION_TEACHER_ACTOR_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_sp_ablation_teacher_actor_heft_reward"
].task_id
TRACKING_BFM_STUDENT_ACTOR_BFM_CRITIC_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_student_actor_bfm_critic_heft_reward"
].task_id
TRACKING_BFM_TEACHER_ACTOR_BFM_CRITIC_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_teacher_actor_bfm_critic_heft_reward"
].task_id
TRACKING_BFM_WBTELEOP_ACTOR_BFM_CRITIC_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_wbteleop_actor_bfm_critic_heft_reward"
].task_id
TRACKING_BFM_WBTELEOP_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_wbteleop_actor_heft_critic_heft_reward"
].task_id
TRACKING_BFM_SPV1_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_spv1_actor_heft_critic_heft_reward"
].task_id
TRACKING_BFM_SPV2_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_spv2_actor_heft_critic_heft_reward"
].task_id
TRACKING_BFM_SPV3_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID = TASK_BY_NAME[
  "tracking_bfm_spv3_actor_heft_critic_heft_reward"
].task_id


def _compose_train(overrides: list[str] | None = None) -> DictConfig:
  overrides = overrides or []
  if GlobalHydra.instance().is_initialized():
    return compose(config_name="train", overrides=overrides)
  with initialize_config_module(version_base=None, config_module="sp_tracking.conf"):
    return compose(config_name="train", overrides=overrides)


def _register_task(task_id: str, overrides: list[str] | None = None) -> None:
  if task_id in list_tasks():
    return
  cfg = _compose_train(overrides)
  play_cfg = _compose_train([*(overrides or []), "task.num_envs=1"])
  register_mjlab_task(
    task_id=task_id,
    env_cfg=build_env_cfg(cfg.task),
    play_env_cfg=build_env_cfg(play_cfg.task),
    rl_cfg=build_agent_cfg(cfg.agent, cfg.task.get("agent_overrides")),
    runner_cls=SpTrackingOnPolicyRunner,
  )


for task_spec in TASK_SPECS:
  _register_task(task_spec.task_id, list(task_spec.hydra_overrides))


__all__ = [
  "TRACKING_BFM_TASK_ID",
  "TRACKING_BFM_SP_TASK_ID",
  "TRACKING_BFM_SP_ABLATION_BFM_ACTOR_TASK_ID",
  "TRACKING_BFM_SP_ABLATION_STUDENT_ACTOR_TASK_ID",
  "TRACKING_BFM_SP_ABLATION_TEACHER_ACTOR_TASK_ID",
  "TRACKING_BFM_STUDENT_ACTOR_BFM_CRITIC_TASK_ID",
  "TRACKING_BFM_TEACHER_ACTOR_BFM_CRITIC_TASK_ID",
  "TRACKING_BFM_WBTELEOP_ACTOR_BFM_CRITIC_TASK_ID",
  "TRACKING_BFM_WBTELEOP_ACTOR_HEFT_CRITIC_TASK_ID",
  "TRACKING_BFM_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_SP_ABLATION_BFM_ACTOR_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_SP_ABLATION_STUDENT_ACTOR_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_SP_ABLATION_TEACHER_ACTOR_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_STUDENT_ACTOR_BFM_CRITIC_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_TEACHER_ACTOR_BFM_CRITIC_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_WBTELEOP_ACTOR_BFM_CRITIC_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_WBTELEOP_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_SPV1_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_SPV2_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID",
  "TRACKING_BFM_SPV3_ACTOR_HEFT_CRITIC_HEFT_REWARD_TASK_ID",
]
