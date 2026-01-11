import torch

def reproduce_win_metrics():
    print("--- Reproducing PPO Win Metrics Logic ---")
    
    # Mocks
    pop_size = 2
    envs_per_agent = 4
    num_envs = pop_size * envs_per_agent
    
    # State Mock
    # 2 Agents. Agent 0: Envs 0-3. Agent 1: Envs 4-7.
    
    # Scenario: Stage 1 (Solo).
    # Active Pods: [0] (Only pod 0 is active and runner).
    # is_runner logic:
    # Pod 0 is Runner. Pods 1,2,3 are Dummy/Inactive or Blocker.
    # In Stage 1 env active_pods=[0].
    
    # is_runner tensor [NumEnvs, 4]
    is_runner = torch.zeros((num_envs, 4), dtype=torch.bool)
    is_runner[:, 0] = True # Pod 0 is always runner in Stage 1
    
    # dones tensor [NumEnvs]
    dones = torch.zeros(num_envs, dtype=torch.bool)
    
    # winners tensor [NumEnvs]
    # -1 = None/Denial, 0 = Team 0 Win, 1 = Team 1 Win
    winners = torch.full((num_envs,), -1, dtype=torch.long)
    
    # Simulate a Win for Agent 0 in Env 0
    dones[0] = True
    winners[0] = 0 # Team 0 Won
    
    # Metric Storage
    # Columns: 8=Wins, 9=Matches, 11=RunnerMatches
    iter_metrics = torch.zeros((pop_size, 20)) 
    
    # === LOGIC FROM PPO.PY (Approx Lines 1930+) ===
    
    # Reshape is_runner [Pop, EnvsPerAgent, 4]
    is_run_reshaped = is_runner.view(pop_size, envs_per_agent, 4)
    
    # Active Pods for this Agent (Simulated)
    active_pods = [0] 
    
    # Reshape Dones [Pop, EnvsPerAgent]
    done_mask = dones.view(pop_size, envs_per_agent)
    
    # Reshape Winners [Pop, EnvsPerAgent]
    w_reshaped = winners.view(pop_size, envs_per_agent)
    
    # 1. Global Win/Match Logic (Lines 1906)
    # wins = (done_mask & (w_reshaped == 0)).float().sum(dim=1)
    # matches = done_mask.float().sum(dim=1)
    
    # iter_metrics[:, 8] += wins
    # iter_metrics[:, 9] += matches
    
    # 2. Per-Pod Logic (Lines 1947+)
    runner_matches = torch.zeros(pop_size)
    blocker_matches = torch.zeros(pop_size)
    winning_matches = torch.zeros(pop_size) # Tracking my own Wins variable
    
    for pod_idx in active_pods:
        # is_run_reshaped[:, :, pod_idx] is 1 if runner, 0 if blocker
        is_r = is_run_reshaped[:, :, pod_idx].float()
        is_b = 1.0 - is_r
        
        # Matches for this pod
        runner_matches += (done_mask.float() * is_r).sum(dim=1)
        blocker_matches += (done_mask.float() * is_b).sum(dim=1)
        
        # Win Calculation Logic in Loop?
        # The code separates Wins (Global) from RunnerMatches (Per Pod).
        # Let's see what happens.
        pass

    # Global Add
    # wins where (Winner == 0) (Team 0).
    wins_global = (done_mask & (w_reshaped == 0)).float().sum(dim=1)
    
    iter_metrics[:, 8] += wins_global
    iter_metrics[:, 11] += runner_matches
    
    print("\n--- Results ---")
    print(f"Agent 0 Wins (Global): {iter_metrics[0, 8]}")
    print(f"Agent 0 Runner Matches: {iter_metrics[0, 11]}")
    
    win_rate = iter_metrics[0, 8] / iter_metrics[0, 11] if iter_metrics[0, 11] > 0 else 0
    print(f"Agent 0 Win Rate: {win_rate}")
    
    if win_rate > 1.0:
        print("!!! FAIL: Win Rate > 1.0 !!!")
    else:
        print("Pass: Logic ok for Single Pod.")

    # === SCENARIO 2: Double Active Pods (Team Stage) ===
    # Active=[0, 1]. Pod 0 Runner, Pod 1 Blocker.
    # Env Winner=0.
    
    print("\n--- Scenario 2: Team Stage (2 Pods) ---")
    active_pods_team = [0, 1]
    is_runner[:, 1] = False # Pod 1 is blocker
    is_run_reshaped = is_runner.view(pop_size, envs_per_agent, 4)
    
    iter_metrics.zero_()
    runner_matches.zero_()
    blocker_matches.zero_()
    
    for pod_idx in active_pods_team:
        is_r = is_run_reshaped[:, :, pod_idx].float()
        is_b = 1.0 - is_r
        
        runner_matches += (done_mask.float() * is_r).sum(dim=1)
        blocker_matches += (done_mask.float() * is_b).sum(dim=1)
        
    wins_global = (done_mask & (w_reshaped == 0)).float().sum(dim=1)
    
    iter_metrics[:, 8] += wins_global
    iter_metrics[:, 11] += runner_matches
    iter_metrics[:, 12] += blocker_matches
    
    print(f"Agent 0 Wins (Global): {iter_metrics[0, 8]}")
    print(f"Agent 0 Runner Matches: {iter_metrics[0, 11]}")
    print(f"Agent 0 Blocker Matches: {iter_metrics[0, 12]}")
    
    win_rate = iter_metrics[0, 8] / iter_metrics[0, 11]
    print(f"Agent 0 Win Rate: {win_rate}") # Should be 1.0 (1 Win / 1 Runner Match)
    
    # === SCENARIO 3: THE BUG ??? ===
    # What if active_pods=[0, 2] (Duel)?
    # Pod 0 Runner, Pod 2 Opponent Runner (but mapped to me?)
    # Stage 1 active_pods=[0].
    # Wait... PPO uses `for pod_idx in active_pods`.
    # `active_pods` comes from `env.config.active_pods`.
    # In Stage 1, active_pods=[0]. Checks out.
    
    # What if `runner_matches` logic is accidentally summing incorrectly?
    # Or `wins` logic?
    
    # Wait! STAGE 1 LOGIC.
    # User said Win Rate ~200%.
    # That means Wins ~ 2 * RunnerMatches.
    # Or RunnerMatches is halved?
    
    # Check if `wins_global` is added multiple times?
    # No, it's outside the loop.
    
    # HYPOTHESIS: Double reset counting?
    # If `collect_rollouts` runs for 512 steps.
    # Env 0 finishes at step 100. Done=True.
    # PPO records metric.
    # Env 0 auto-resets. Done=False next step.
    # Logic holds.
    
    # HYPOTHESIS: `active_pods` in Stage 1 includes [0, 2] or something?
    # If Stage 1 meant to be Solo, but config says active_pods=[0, 2] (Duel setup)?
    # Then I have 2 pods.
    # Pod 0 is Runner. Pod 2 is... also Runner (Enemy).
    # If I control Pod 2 (Self-Play/League setup logic leakage)?
    # In PPO, `active_pods` determines loop.
    # If active_pods=[0, 2].
    # loop pod 0: is_r=1. runner_matches += 1.
    # loop pod 2: is_r=1. runner_matches += 1.
    # Total runner_matches = 2.
    # Wins = 1 (Global episode win).
    # Win Rate = 1/2 = 50%.
    # Wait, 200% means Wins=2, Matches=1.
    
    # How to get Wins=2?
    # Wins is calculated once globally.
    
    # UNLESS... `active_pods` is empty? Division by zero? No, float.
    
    # check line 629 in PPO (previous view)
    # p['win_rate'] = float(wins) / float(runner_matches)
    
    # Maybe `wins` variable in p dictionary is accumulating DOUBLE?
    # `p['wins']` comes from `_gen_metrics`.
    # `_gen_metrics` sums `_iter_metrics` over epochs?
    # No, PPO resets `_iter_metrics` per iteration.
    
    # What if `wins` includes "Step Wins"? 
    # i.e. "Win Reward" triggers multple times?
    # Line 1907: `wins = (done_mask & (w_reshaped==0))...`
    # It counts Dones. Dones are boolean.
    
    pass

if __name__ == "__main__":
    reproduce_win_metrics()
