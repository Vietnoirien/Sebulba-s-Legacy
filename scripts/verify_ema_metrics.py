
import torch
from simulation.env import PodRacerEnv, STAGE_SOLO, STAGE_DUEL, STAGE_TEAM, TIMEOUT_STEPS

def verify_ema_metrics():
    print("Locked & Loaded: Verifying EMA Metrics robustness against Timeouts...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 10
    
    # --- Test 1: Stage 1 (Solo/Duel) Timeout Handling ---
    print("\n--- Phase 1: Stage 1 (Duel) Metrics ---")
    env = PodRacerEnv(num_envs, device=device, start_stage=STAGE_DUEL) # Duel checks wins/losses
    
    # Initial State
    print(f"Initial Recent Episodes: {env.stage_metrics['recent_episodes']}")
    print(f"Initial Recent Wins: {env.stage_metrics['recent_wins']}")
    
    # Force Timeout
    print(f"Simulating {TIMEOUT_STEPS} steps of inactivity...")
    actions = torch.zeros((num_envs, 4, 4), device=device)
    
    for _ in range(TIMEOUT_STEPS):
        _, dones, _ = env.step(actions, reward_weights=None)
        
    assert dones.all(), "Environment did not timeout as expected!"
    
    # Check Metrics
    rec_episodes = env.stage_metrics['recent_episodes']
    rec_wins = env.stage_metrics['recent_wins']
    
    print(f"Post-Timeout Recent Episodes: {rec_episodes}")
    print(f"Post-Timeout Recent Wins: {rec_wins}")
    
    # Verify
    # We expect 10 episodes (since 10 envs timed out).
    # We expect 0 wins (all timed out).
    
    assert rec_episodes == num_envs, f"Expected {num_envs} episodes, got {rec_episodes}"
    assert rec_wins == 0, f"Expected 0 wins, got {rec_wins}"
    
    loss_rate = (rec_episodes - rec_wins) / rec_episodes
    print(f"Implied Win Rate: {rec_wins / rec_episodes:.2f}")
    print(f"Implied Loss Rate (Timeouts included): {loss_rate:.2f}")
    
    if rec_wins == 0 and rec_episodes > 0:
        print(">> SUCCESS: Timeouts count as Non-Wins (Losses) in Stage 1.")
    else:
        print(">> FAILURE: Metrics do not reflect timeouts correctly.")

    # --- Test 2: Stage 2 (Team) Timeout Handling ---
    print("\n--- Phase 2: Stage 2 (Team) Metrics ---")
    env = PodRacerEnv(num_envs, device=device, start_stage=STAGE_TEAM)
    
    # Reset Metrics manually just in case init didn't clean (it should have)
    env.stage_metrics['recent_episodes'] = 0
    env.stage_metrics['recent_wins'] = 0
    
    # Force Timeout
    print(f"Simulating {TIMEOUT_STEPS} steps of inactivity...")
    for _ in range(TIMEOUT_STEPS):
        _, dones, _ = env.step(actions, reward_weights=None)
        
    assert dones.all()
    
    rec_episodes = env.stage_metrics['recent_episodes']
    rec_wins = env.stage_metrics['recent_wins']
    
    print(f"Post-Timeout Recent Episodes: {rec_episodes}")
    print(f"Post-Timeout Recent Wins: {rec_wins}")
    
    assert rec_episodes == num_envs, f"Expected {num_envs} episodes, got {rec_episodes}"
    assert rec_wins == 0
    
    print(">> SUCCESS: Timeouts count as Non-Wins (Losses) in Stage 2.")
    
    print("\nVerification Complete: System is SOTA compliant (Timeouts = Losses).")

if __name__ == "__main__":
    verify_ema_metrics()
