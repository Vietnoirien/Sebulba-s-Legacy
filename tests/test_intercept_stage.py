
import torch
from config import CurriculumConfig, EnvConfig
from training.curriculum.stages import InterceptStage
from simulation.env import PodRacerEnv, STAGE_INTERCEPT

def test_intercept_stage_config():
    """Verify InterceptStage returns correct EnvConfig."""
    c_config = CurriculumConfig()
    stage = InterceptStage(c_config)
    
    env_config = stage.get_env_config()
    
    assert env_config.mode_name == "intercept"
    assert env_config.active_pods == [0, 2]
    assert env_config.bot_pods == [2]
    assert env_config.fixed_roles == {0: 0, 2: 1} # 0=Blocker, 1=Runner
    
def test_intercept_stage_env_init():
    """Verify Environment respects fixed_roles from InterceptStage."""
    c_config = CurriculumConfig()
    stage = InterceptStage(c_config)
    env_config = stage.get_env_config()
    
    # Initialize Env
    # Need to mock device? VectorizedEnvironment needs some mocks or real cpu run.
    # We use minimal config.
    
    # We need a fake trainer or manual env init
    # Let's trust VectorizedEnvironment works if we pass config.
    
    # Fake config
    env = PodRacerEnv(num_envs=2, device="cpu")
    # Manually set config (hack for test since we didn't use trainer)
    env.set_stage(STAGE_INTERCEPT, env_config, reset_env=True)
    
    # Reset
    env.reset()
    
    # Check Active Pods
    # Pod 0 and 2 should be active (pos != -100000)
    # Pod 1 and 3 should be inactive
    
    pos = env.physics.pos
    # Check X coord > -50000 (valid spawn is near 0 or track)
    assert torch.all(pos[:, 0, 0] > -50000)
    assert torch.all(pos[:, 2, 0] > -50000)
    assert torch.all(pos[:, 1, 0] < -50000) # Inactive
    assert torch.all(pos[:, 3, 0] < -50000) # Inactive
    
    # Check Roles
    # is_runner: [N, 4] boolean
    # Pod 0 should be False (Blocker)
    # Pod 2 should be True (Runner)
    
    assert torch.all(env.is_runner[:, 0] == False)
    assert torch.all(env.is_runner[:, 2] == True)
    
    print("Intercept Stage Config & Role Verification Passed!")

if __name__ == "__main__":
    test_intercept_stage_config()
    test_intercept_stage_env_init()
