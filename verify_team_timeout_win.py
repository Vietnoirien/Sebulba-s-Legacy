import torch
from simulation.env import PodRacerEnv, STAGE_TEAM

def test_team_timeout_win():
    print("Initializing Env in STAGE_TEAM...")
    env = PodRacerEnv(num_envs=2, device='cuda', start_stage=STAGE_TEAM)
    
    # Configure Team Stage active pods just in case
    # STAGE_TEAM usually implies all 4 pods active
    
    # Reset to clear state
    env.reset()
    
    # Scenario 1: Team 0 (Pod 0) Times Out
    print("\n--- Scenario 1: Team 0 Pod 0 Timeout ---")
    env.timeouts[:] = 100 # Reset timeouts
    env.timeouts[0, 0] = 1 # Set Pod 0 to 1 step remaining
    
    # Step
    actions = torch.zeros((2, 4, 4), device='cuda')
    env.step(actions, None)
    
    # Check Result
    # Expect: Done=True, Winner=1 (Team 1 wins because Team 0 timed out)
    done_0 = env.dones[0].item()
    winner_0 = env.winners[0].item()
    
    print(f"Env 0 Done: {done_0}")
    print(f"Env 0 Winner: {winner_0} (Expected: 1)")
    
    if done_0 and winner_0 == 1:
        print("PASS: Team 0 Timeout resulted in Team 1 Win.")
    else:
        print("FAIL: Team 0 Timeout did NOT result in Team 1 Win.")

    # Scenario 2: Team 1 (Pod 2) Times Out
    print("\n--- Scenario 2: Team 1 Pod 2 Timeout ---")
    # Reset logic might have fired if done was true, but let's force state
    # We use Env 1 for this test to avoid reset noise if any
    env.timeouts[1, :] = 100
    env.timeouts[1, 2] = 1 # Set Pod 2 (Team 1) to 1 step
    
    env.step(actions, None)
    
    done_1 = env.dones[1].item()
    winner_1 = env.winners[1].item()
    
    print(f"Env 1 Done: {done_1}")
    print(f"Env 1 Winner: {winner_1} (Expected: 0)")
    
    if done_1 and winner_1 == 0:
        print("PASS: Team 1 Timeout resulted in Team 0 Win.")
    else:
        print("FAIL: Team 1 Timeout did NOT result in Team 0 Win.")

if __name__ == "__main__":
    try:
        test_team_timeout_win()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
