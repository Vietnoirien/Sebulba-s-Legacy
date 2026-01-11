
import torch

def test_reward_logic():
    device = 'cpu'
    num_envs = 1
    
    # 1. Setup State
    # Pod 0: Runner (True)
    # Pod 2: Runner (True)
    is_runner = torch.tensor([[True, False, True, False]], device=device) # [B, 4]
    
    # Fake Collisions [B, 4, 4]
    collisions = torch.zeros((num_envs, 4, 4), device=device)
    # Collision between 0 and 2 (Runner vs Runner)
    collisions[:, 0, 2] = 100.0 
    collisions[:, 2, 0] = 100.0
    
    # Rewards
    rewards_indiv = torch.zeros((num_envs, 4), device=device)
    blocker_damage_metric = torch.zeros((num_envs, 4), device=device)
    
    w_col_run = 0.5
    w_col_block = 1000.0
    w_col_mate = 2.0
    S_VEL = 0.001
    
    # Mock Physics Vel
    vel = torch.zeros((num_envs, 4, 2), device=device)
    
    print("--- Starting Test ---")
    print(f"Is Runner: {is_runner}")
    print(f"Collision 0-2: {collisions[:, 0, 2]}")
    
    # --- LOGIC COPIED FROM ENV.PY (Lines 889-924) ---
    runner_velocity_metric = torch.zeros((num_envs, 4), device=device)
    
    for i in range(4):
        team = i // 2
        enemy_team = 1 - team
        enemy_indices = [2*enemy_team, 2*enemy_team + 1]
        
        is_run = is_runner[:, i] # [B]
        
        v_mag = torch.norm(vel[:, i], dim=1)
        runner_velocity_metric[:, i] = v_mag * is_run.float() * S_VEL 
        
        impact_e1 = collisions[:, i, enemy_indices[0]]
        impact_e2 = collisions[:, i, enemy_indices[1]]
        total_impact = impact_e1 + impact_e2
        
        # 1. Runner Penalty
        runner_pen = -w_col_run * total_impact
        # rewards_indiv[:, i] += runner_pen * is_run.float() 
        # (Skip applying penalty to check blocker reward isolation)
        
        # 2. Blocker Bonus
        is_block = ~is_run
        enemy_runner_mask = is_runner[:, enemy_indices] # [B, 2]
        
        bonus = torch.zeros(num_envs, device=device)
        bonus += impact_e1 * enemy_runner_mask[:, 0].float()
        bonus += impact_e2 * enemy_runner_mask[:, 1].float()
        
        blocker_damage_metric[:, i] = bonus * is_block.float()
        
        blocker_reward = w_col_block * bonus
        
        # The Line in Question
        rewards_indiv[:, i] += blocker_reward * is_block.float()
        
        print(f"Pod {i} | Role: {'Runner' if is_run.item() else 'Blocker'} | Impact: {total_impact.item()} | Bonus Calc: {bonus.item()} | Blocker Reward: {(blocker_reward * is_block.float()).item()}")

    print("--- End Test ---")
    print(f"Rewards Indiv: {rewards_indiv}")

if __name__ == "__main__":
    test_reward_logic()
