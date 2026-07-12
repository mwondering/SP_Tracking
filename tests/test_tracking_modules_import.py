def test_tracking_mdp_modules_import() -> None:
  import sp_tracking.tasks.tracking.mdp.multi_command_largedataset as large_dataset
  import sp_tracking.tasks.tracking.mdp.multi_commands as multi_commands
  import sp_tracking.tasks.tracking.mdp.sp as sp_mdp

  assert hasattr(multi_commands, "MotionCommandCfg")
  assert hasattr(large_dataset, "MotionCommandCfg")
  assert hasattr(sp_mdp, "SP_REQUIRED_BODY_NAMES")


def test_tracking_rl_modules_import() -> None:
  from sp_tracking.tasks.tracking.rl import SpTrackingOnPolicyRunner
  from sp_tracking.tasks.tracking.rl.ppo import SparseTrackSplitLrPPO

  assert SpTrackingOnPolicyRunner is not None
  assert SparseTrackSplitLrPPO is not None
