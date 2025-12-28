import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import time
import os
import asyncio
import json
import aiohttp
import uuid
from typing import List, Tuple, Dict, Optional
import copy

# Import Env and Constants
from simulation.env import (
    PodRacerEnv, 
    RW_WIN, RW_LOSS, RW_CHECKPOINT, RW_CHECKPOINT_SCALE, 
    RW_VELOCITY, RW_COLLISION_RUNNER, RW_COLLISION_BLOCKER, 
    RW_STEP_PENALTY, RW_ORIENTATION, RW_WRONG_WAY, RW_COLLISION_MATE,
    DEFAULT_REWARD_WEIGHTS
)
from models.deepsets import PodAgent
from training.self_play import LeagueManager
from training.normalization import RunningMeanStd
from training.evolution import calculate_novelty, fast_non_dominated_sort, calculate_crowding_distance
from training.rnd import RNDModel
from config import *

# Hyperparameters
LR = 1e-4
GAMMA = 0.994
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
DIV_COEF = 0.05 # Role Regularization Coefficient
TOTAL_TIMESTEPS = 2_000_000_000
NUM_ENVS = 4096
NUM_STEPS = 256
# PBT Settings
POP_SIZE = 32
ENVS_PER_AGENT = NUM_ENVS // POP_SIZE # 128
assert NUM_ENVS % POP_SIZE == 0, "NUM_ENVS must be divisible by POP_SIZE"
EVOLVE_INTERVAL = 2 # Updates between evolutions

# Batch Size per Agent
BATCH_SIZE = 2 * ENVS_PER_AGENT * NUM_STEPS 
MINIBATCH_SIZE = 16384 # 12GB VRAM can handle this easily
UPDATE_EPOCHS = 4
REPORT_INTERVAL = 1 

class PPOTrainer:

    def __init__(self, device='cuda', logger_callback=None):
        self.device = torch.device(device)
        self.env = PodRacerEnv(NUM_ENVS, device=device)
        self.logger_callback = logger_callback
        
        # Population Initialization
        self.population = []
        self.generation = 0
        self.iteration = 0
        
        # Reward Tensors [4096, 11]
        self.reward_weights_tensor = torch.zeros((NUM_ENVS, 12), device=self.device)
        
        # Normalization
        self.rms_self = RunningMeanStd((14,), device=self.device)
        self.rms_ent = RunningMeanStd((13,), device=self.device)
        self.rms_cp = RunningMeanStd((6,), device=self.device)
        
        # RND Intrinsic Curiosity
        # Input: Normalized Self Obs (14)
        self.rnd = RNDModel(input_dim=14, device=self.device)
        self.rnd_coef = 0.01 # PPO Intrinsic Coefficient
        
        # Reward Normalization
        self.rms_ret = RunningMeanStd((1,), device=self.device)
        self.returns_buffer = torch.zeros((NUM_ENVS, 4), device=self.device)
        
        for i in range(POP_SIZE):
            agent = PodAgent().to(self.device)
            optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
            
            # Initial Reward Config (Clone Default)
            # Add some noise for initial diversity?
            weights = DEFAULT_REWARD_WEIGHTS.copy()
            # Randomize slightly?
            if i > 0:
                 # Mutate orientation and velocity slightly
                 weights[RW_ORIENTATION] *= random.uniform(0.8, 1.2)
                 weights[RW_VELOCITY] *= random.uniform(0.8, 1.2)
            
            self.population.append({
                'id': i,
                'type': 'exploiter' if i >= 28 else 'main', # 4 Explicit Exploiters
                'agent': agent,
                'optimizer': optimizer,
                'weights': weights, # Python Dict for mutation logic
                'lr': LR,
                'ent_coef': ENT_COEF, # Individual Entropy Coefficient
                'clip_range': CLIP_RANGE, # Individual Clip Range
                'laps_score': 0,
                'checkpoints_score': 0,
                'reward_score': 0.0,
                'efficiency_score': 999.0,
                'wins': 0, # Track wins for League stage
                'matches': 0, # Track completed matches
                'max_streak': 0,
                'total_cp_steps': 0,
                'total_cp_hits': 0,
                'avg_steps': 0.0,
                'avg_runner_vel': 0.0,
                'avg_blocker_dmg': 0.0,
                
                # --- GA Robustness (EMA Stats) ---
                'ema_efficiency': None, # Lower is better? We will invert or handle logic.
                'ema_consistency': None, # Checkpoint Score
                'ema_wins': None,
                'ema_laps': None,
                'ema_runner_vel': None,
                'ema_blocker_dmg': None,
                'ema_dist': None, # Novelty Distance
                
                # --- Behavior Characterization ---
                # Buffer to accumulate [Speed, Steering, MateDist] per step: [SumSpeed, SumSteer, SumDist, Count]
                'behavior_buffer': torch.zeros(4, device=self.device)
            })
            
            # Fill Tensor
            start_idx = i * ENVS_PER_AGENT
            end_idx = start_idx + ENVS_PER_AGENT
            for k, v in weights.items():
                self.reward_weights_tensor[start_idx:end_idx, k] = v

        # Default Pointer for API compatibility (Leader)
        self.leader_idx = 0

        # League & Opponent
        self.league = LeagueManager()
        self.opponent_agent = PodAgent().to(self.device)
        
        self.match_id = str(uuid.uuid4())
        self.active_model_name = "scratch"
        self.curriculum_mode = "auto" 
        
        # Telemetry
        # Track 2 distinct streams for visualization continuity
        # Updated to track 32 streams (One per agent) to ensure "Playlist" saturation
        self.telemetry_env_indices = [i * 128 for i in range(32)] 
        self.stats_interval = 100
        
        # EMA Alpha (Smoothing Factor)
        # 0.3 means 30% new, 70% old. 
        # ~3 generations memory.
        self.ema_alpha = 0.3

        # Dynamic Hyperparameters
        self.current_num_steps = 256 # Start with Stage 0 default
        self.current_evolve_interval = 2 # Start with Stage 0 default
        self.current_active_pods_count = 1 # Start with Stage 0 default
        self.agent_batches = [] 
        
        # Difficulty Adjustment State
        self.failure_streak = 0
        self.grad_consistency_counter = 0
        self.team_spirit = 0.0 # Blending factor for rewards (0.0=Selfish, 1.0=Cooperative)
        self.current_win_rate = 0.0 # Persistent Win Rate for Telemetry
        
        # TRANSITION CONFIGURATION
        self.curriculum_config = {
             # Stage 0 -> 1
             "solo_efficiency_threshold": STAGE_SOLO_EFFICIENCY_THRESHOLD,
             "solo_consistency_threshold": STAGE_SOLO_CONSISTENCY_THRESHOLD,
             
             # Stage 1 -> 2
             "duel_consistency_wr": 0.82,
             "duel_absolute_wr": 0.84,
             "duel_consistency_checks": 5,
             
             # Stage 2 -> 3
             "team_consistency_wr": 0.85,
             "team_absolute_wr": 0.88,
             "team_consistency_checks": 5
        }
        
        # Allocate Initial Buffers
        self.allocate_buffers()

    def allocate_buffers(self):
        """Allocates or Re-allocates batch buffers based on current_num_steps"""
        active_pods = self.get_active_pods()
        self.current_active_pods_count = len(active_pods)
        self.log(f"Allocating buffers for {self.current_num_steps} steps per iteration. Active Pods: {self.current_active_pods_count}")
        
        # Decide Active Pods Max (Assume max 2 for buffer sizing to be safe, or separate?)
        # Buffers are per AGENT.
        # Envs per agent = 128.
        # Max pods per env = 2 (League).
        
        num_active_pods = self.current_active_pods_count
            
        # Total batch dimension = Steps * EnvsPerAgent * NumActivePods
        num_active_per_agent_step = num_active_pods * ENVS_PER_AGENT
        
        self.agent_batches = []
        for _ in range(POP_SIZE):
             self.agent_batches.append({
                 'self_obs': torch.zeros((self.current_num_steps, num_active_per_agent_step, 14), device=self.device),
                 'teammate_obs': torch.zeros((self.current_num_steps, num_active_per_agent_step, 13), device=self.device),
                 'enemy_obs': torch.zeros((self.current_num_steps, num_active_per_agent_step, 2, 13), device=self.device),
                 'cp_obs': torch.zeros((self.current_num_steps, num_active_per_agent_step, 6), device=self.device),
                 'actions': torch.zeros((self.current_num_steps, num_active_per_agent_step, 4), device=self.device),
                 'logprobs': torch.zeros((self.current_num_steps, num_active_per_agent_step), device=self.device),
                 'rewards': torch.zeros((self.current_num_steps, num_active_per_agent_step), device=self.device),
                 'dones': torch.zeros((self.current_num_steps, num_active_per_agent_step), device=self.device),
                 'values': torch.zeros((self.current_num_steps, num_active_per_agent_step), device=self.device)
             })
    
    def log(self, msg):
        print(msg)
        if self.logger_callback:
            self.logger_callback(msg)

    @property
    def agent(self):
        return self.population[self.leader_idx]['agent']

    @property
    def reward_config(self):
        # Compatibility property returning Leader's config
        w = self.population[self.leader_idx]['weights']
        return {
             "tau": 0.0, "beta": 0.0, "weights": {
                 "win": w[RW_WIN], "loss": w[RW_LOSS], 
                 "checkpoint": w[RW_CHECKPOINT], "checkpoint_scale": w[RW_CHECKPOINT_SCALE],
                 "velocity": w[RW_VELOCITY], "collision_runner": w[RW_COLLISION_RUNNER],
                 "collision_blocker": w[RW_COLLISION_BLOCKER], "step_penalty": w[RW_STEP_PENALTY],
                 "orientation": w[RW_ORIENTATION], "wrong_way_alpha": w[RW_WRONG_WAY],
                 "collision_mate": w[RW_COLLISION_MATE]
             }
        }

    async def send_telemetry(self, step, fps_phys, fps_train, reward, win_rate, env_idx=0):
        # Identify Agent
        agent_idx = env_idx // ENVS_PER_AGENT
        pop_member = self.population[agent_idx]
        
        # Extract state
        try:
            pods = []
            for i in range(4):
                pos = self.env.physics.pos[env_idx, i]
                angle = self.env.physics.angle[env_idx, i]
                pods.append({
                    "id": i, "team": i // 2, "x": float(pos[0]), "y": float(pos[1]),
                    "angle": float(angle), "boost": 1, "shield": 0 
                })
            
            checkpoints = []
            n_cps = self.env.num_checkpoints[env_idx]
            for i in range(n_cps):
                cp = self.env.checkpoints[env_idx, i]
                checkpoints.append({"x": float(cp[0]), "y": float(cp[1]), "id": i, "radius": 600})

            payload = {
                "type": "telemetry",
                "step": step,
                "match_id": self.match_id,
                "stats": {
                    "fps_physics": float(fps_phys),
                    "fps_training": float(fps_train),
                    "reward_mean": float(reward),
                    "win_rate": float(win_rate),
                    "active_model": f"Generaton {self.generation} | Agent {agent_idx}",
                    "curriculum_stage": int(self.env.curriculum_stage),
                    "generation": self.generation,
                    "agent_id": agent_idx
                },
                "race_state": {"pods": pods, "checkpoints": checkpoints}
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post('http://localhost:8000/api/telemetry', json=payload)
        except Exception:
            pass

    def check_curriculum(self):
        # if self.curriculum_mode == "manual": return # Removed to allow logging

        metrics = self.env.stage_metrics
        stage = self.env.curriculum_stage
        
        if stage == STAGE_SOLO:
            # Strategy A: Proficiency & Consistency
            # Thresholds: Eff < 30.0, Consistency > 2000.0 (Sum of CPs)
            
            # Top 5 Agents by Consistency (Sum of CPs)
            # Consistency is stored as 'ema_consistency'
            sorted_by_consistency = sorted(self.population, key=lambda x: x.get('ema_consistency') or 0.0, reverse=True)
            elites = sorted_by_consistency[:5]
            
            avg_eff = np.mean([p.get('efficiency_score') if p.get('efficiency_score') is not None else 999.0 for p in elites])
            avg_cons = np.mean([p.get('ema_consistency') if p.get('ema_consistency') is not None else 0.0 for p in elites])
            
            # Log progress periodically
            if self.iteration % 10 == 0:
                 self.log(f"Stage 0 Status: Top 5 Avg Eff {avg_eff:.1f} (Goal < {STAGE_SOLO_EFFICIENCY_THRESHOLD}), Cons {avg_cons:.1f} (Goal > {STAGE_SOLO_CONSISTENCY_THRESHOLD})")
            
            if avg_eff < STAGE_SOLO_EFFICIENCY_THRESHOLD and avg_cons > STAGE_SOLO_CONSISTENCY_THRESHOLD:
                self.log(f">>> GRADUATION: Top Agents Avg Eff {avg_eff:.1f}, Cons {avg_cons:.1f} <<<")
                if self.curriculum_mode == "auto":
                    self.env.curriculum_stage = STAGE_DUEL
                    self.env.bot_difficulty = 0.0

        elif stage == STAGE_DUEL:
            # Dynamic Difficulty
            # Check every 1k EPISODES (including timeouts)
            rec_episodes = metrics.get("recent_episodes", 0)
            if rec_episodes == 0:
                 # Fallback for old env code or if not populated yet
                 rec_episodes = metrics["recent_games"]

            if rec_episodes > 1000: 
                rec_wins = metrics["recent_wins"]
                rec_games = metrics["recent_games"] # Valid finished games
                
                if rec_episodes > 0:
                    rec_wr = rec_wins / rec_episodes
                else:
                    rec_wr = 0.0 # All Timeouts = 0% Win Rate
                
                self.current_win_rate = rec_wr # Update global tracker
                
                # Reset Recent
                metrics["recent_games"] = 0
                metrics["recent_wins"] = 0
                metrics["recent_episodes"] = 0
                
                # Wins, Losses, Timeouts
                rec_losses = rec_games - rec_wins
                rec_timeouts = rec_episodes - rec_games
                
                self.log(f"Stage 1 Check: Recent WR {rec_wr*100:.1f}% | Wins: {rec_wins} | Losses: {rec_losses} | Timeouts: {rec_timeouts} | Diff: {self.env.bot_difficulty:.2f}")
                
                if rec_wr < 0.30:
                    # Critical Failure: Immediate Regression
                    if self.curriculum_mode == "auto":
                        self.env.bot_difficulty = max(0.0, self.env.bot_difficulty - 0.05)
                        self.log(f"-> Critical Regression (WR < 30%): Decreasing Bot Difficulty to {self.env.bot_difficulty:.2f}")
                    else:
                        self.log(f"-> Critical Regression (WR < 30%): [Manual] Suggested Diff: {max(0.0, self.env.bot_difficulty - 0.05):.2f}")
                    
                    self.failure_streak = 0 # Reset streak modification
                    
                elif rec_wr < 0.40:
                    # Warning Zone: Regression only if persistent
                    self.failure_streak += 1
                    self.log(f"-> Warning Zone (WR < 40%): Streak {self.failure_streak}/2")
                    
                    if self.failure_streak >= 2:
                        if self.curriculum_mode == "auto":
                            self.env.bot_difficulty = max(0.0, self.env.bot_difficulty - 0.05)
                            self.log(f"-> Persistent Failure: Decreasing Bot Difficulty to {self.env.bot_difficulty:.2f}")
                        else:
                            self.log(f"-> Persistent Failure: [Manual] Suggested Diff: {max(0.0, self.env.bot_difficulty - 0.05):.2f}")
                        self.failure_streak = 0
                
                else:
                    # Winning enough to stabilize or progress
                    self.failure_streak = 0
                    

                    new_diff = self.env.bot_difficulty
                    if rec_wr > 0.99:
                        new_diff = min(1.0, self.env.bot_difficulty + 0.50)
                        msg = "Insane Turbo"
                    elif rec_wr > 0.98:
                        new_diff = min(1.0, self.env.bot_difficulty + 0.20)
                        msg = "Super Turbo"
                    elif rec_wr > 0.90:
                        new_diff = min(1.0, self.env.bot_difficulty + 0.10)
                        msg = "Turbo"
                    elif rec_wr > 0.70:
                        new_diff = min(1.0, self.env.bot_difficulty + 0.05)
                        msg = "Standard"
                    else:
                        msg = None

                    if self.env.bot_difficulty < 1.0:
                        if msg:
                             if self.curriculum_mode == "auto":
                                 self.env.bot_difficulty = new_diff
                                 self.log(f"-> {msg}: Increasing Bot Difficulty to {self.env.bot_difficulty:.2f}")
                             else:
                                 self.log(f"-> {msg}: [Manual] Suggested Diff: {new_diff:.2f}")
                    else:
                         # Max difficulty reached. Show status.
                         cons_wr = self.curriculum_config["duel_consistency_wr"]
                         abs_wr = self.curriculum_config["duel_absolute_wr"]
                         cons_checks = self.curriculum_config["duel_consistency_checks"]
                         
                         streak = self.grad_consistency_counter
                         if rec_wr > cons_wr:
                             streak += 1
                         else:
                             streak = 0
                             
                         self.log(f"-> Max Difficulty. Next Stage Req: WR >= {abs_wr:.2f} OR (WR > {cons_wr:.2f} Streak {streak}/{cons_checks})")
                
                # Graduation Check
                # SOTA Update: Smoother transition.
                # Threshold: > 0.84 ONCE or > 0.82 for 5 sequential checks at difficulty 1.0.
                if self.env.bot_difficulty >= 1.0:
                    should_graduate = False
                    reason = ""
                    
                    cons_wr = self.curriculum_config["duel_consistency_wr"]
                    abs_wr = self.curriculum_config["duel_absolute_wr"]
                    cons_checks = self.curriculum_config["duel_consistency_checks"]

                    if rec_wr > cons_wr:
                        self.grad_consistency_counter += 1
                    else:
                        self.grad_consistency_counter = 0

                    if rec_wr >= abs_wr and self.grad_consistency_counter >= 2:
                        should_graduate = True
                        reason = f"WR {rec_wr:.2f} >= {abs_wr}"
                    elif self.grad_consistency_counter >= cons_checks:
                        should_graduate = True
                        reason = f"WR > {cons_wr} for {cons_checks} checks (Last: {rec_wr:.2f})"
                        
                    if should_graduate:
                         self.log(f">>> UPGRADING TO STAGE 2: TEAM ({reason}) <<<")
                         if self.curriculum_mode == "auto":
                             self.env.curriculum_stage = STAGE_TEAM
                             self.env.bot_difficulty = 0.0 # Reset difficulty for new stage
                             
                             # Reset Metrics for new stage
                             self.env.stage_metrics["recent_games"] = 0
                             self.env.stage_metrics["recent_wins"] = 0
                             self.env.stage_metrics["recent_episodes"] = 0
                         
        elif stage == STAGE_TEAM:
            # Dynamic Difficulty (Same Logic as Duel but for 2v2)
            rec_episodes = metrics.get("recent_episodes", 0)
            if rec_episodes == 0: rec_episodes = metrics["recent_games"]

            if rec_episodes > 1000: 
                rec_wins = metrics["recent_wins"]
                rec_games = metrics["recent_games"]
                
                if rec_episodes > 0:
                    rec_wr = rec_wins / rec_episodes
                else:
                    rec_wr = 0.0
                
                self.current_win_rate = rec_wr # Update global tracker
                
                # Reset Recent
                metrics["recent_games"] = 0
                metrics["recent_wins"] = 0
                metrics["recent_episodes"] = 0
                
                # Wins, Losses, Timeouts
                rec_losses = rec_games - rec_wins
                rec_timeouts = rec_episodes - rec_games

                self.log(f"Stage 2 (Team) Check: Recent WR {rec_wr*100:.1f}% | Wins: {rec_wins} | Losses: {rec_losses} | Timeouts: {rec_timeouts} | Diff: {self.env.bot_difficulty:.2f}")
                
                if rec_wr < 0.30:
                    if self.curriculum_mode == "auto":
                        self.env.bot_difficulty = max(0.0, self.env.bot_difficulty - 0.05)
                        self.log(f"-> Critical Regression (WR < 30%): Decreasing Bot Difficulty to {self.env.bot_difficulty:.2f}")
                    else:
                        self.log(f"-> Critical Regression (WR < 30%): [Manual] Suggested Diff: {max(0.0, self.env.bot_difficulty - 0.05):.2f}")
                    self.failure_streak = 0
                    
                elif rec_wr < 0.40:
                    self.failure_streak += 1
                    self.log(f"-> Warning Zone (WR < 40%): Streak {self.failure_streak}/2")
                    
                    if self.failure_streak >= 2:
                        self.env.bot_difficulty = max(0.0, self.env.bot_difficulty - 0.05)
                        self.failure_streak = 0
                        self.log(f"-> Persistent Failure: Decreasing Bot Difficulty to {self.env.bot_difficulty:.2f}")
                
                else:
                    self.failure_streak = 0
                    
                    new_diff = self.env.bot_difficulty
                    if rec_wr > 0.98:
                        new_diff = min(1.0, self.env.bot_difficulty + 0.20)
                        msg = "Super Turbo"
                    elif rec_wr > 0.90:
                        new_diff = min(1.0, self.env.bot_difficulty + 0.10)
                        msg = "Turbo"
                    elif rec_wr > 0.70:
                        new_diff = min(1.0, self.env.bot_difficulty + 0.05)
                        msg = "Standard"
                    else:
                        msg = None
                    
                    if self.env.bot_difficulty < 1.0:
                        if msg:
                            if self.curriculum_mode == "auto":
                                self.env.bot_difficulty = new_diff
                                self.log(f"-> {msg}: Increasing Bot Difficulty to {self.env.bot_difficulty:.2f}")
                            else:
                                self.log(f"-> {msg}: [Manual] Suggested Diff: {new_diff:.2f}")
                    else:
                         # Max difficulty reached. Show status.
                         cons_wr = self.curriculum_config["team_consistency_wr"]
                         abs_wr = self.curriculum_config["team_absolute_wr"]
                         cons_checks = self.curriculum_config["team_consistency_checks"]
                         
                         streak = self.grad_consistency_counter
                         if rec_wr > cons_wr:
                             streak += 1
                         else:
                             streak = 0
                             
                         self.log(f"-> Max Difficulty. Next Stage Req: WR >= {abs_wr:.2f} OR (WR > {cons_wr:.2f} Streak {streak}/{cons_checks})")
                
                # Graduation Check (To League)
                cons_wr = self.curriculum_config["team_consistency_wr"]
                abs_wr = self.curriculum_config["team_absolute_wr"]
                cons_checks = self.curriculum_config["team_consistency_checks"]
                
                if rec_wr > cons_wr: # Slightly higher bar for 2v2 mastery
                    self.grad_consistency_counter += 1
                else:
                    self.grad_consistency_counter = 0
                    
                if self.env.bot_difficulty >= 1.0:
                    should_graduate = False
                    reason = ""
                    
                    if rec_wr >= abs_wr:
                        should_graduate = True
                        reason = f"WR {rec_wr:.2f} >= {abs_wr}"
                    elif self.grad_consistency_counter >= cons_checks:
                        should_graduate = True
                        reason = f"WR > {cons_wr} for {cons_checks} checks (Last: {rec_wr:.2f})"
                        
                    if should_graduate:
                         self.log(f">>> UPGRADING TO STAGE 3: LEAGUE ({reason}) <<<")
                         if self.curriculum_mode == "auto":
                             self.env.curriculum_stage = STAGE_LEAGUE
                             
                             # Reset Metrics for new stage
                             self.env.stage_metrics["recent_games"] = 0
                             self.env.stage_metrics["recent_wins"] = 0
                             self.env.stage_metrics["recent_episodes"] = 0
        
        # --- Team Spirit Update ---
        if stage < STAGE_TEAM:
            self.team_spirit = 0.0
        elif stage == STAGE_TEAM:
            # Anneal based on Difficulty or WR?
            # Let's link it to Difficulty (which proxies capability).
            # Diff 0.0 -> Spirit 0.0
            # Diff 1.0 -> Spirit 0.5
            target_spirit = self.env.bot_difficulty * 0.5
            self.team_spirit = target_spirit
        else: # LEAGUE
            self.team_spirit = 1.0
            
        # Update Step Penalty based on new difficulty/stage
        self.update_step_penalty_annealing()

    def get_active_pods(self):
        stage = self.env.curriculum_stage
        if stage == STAGE_SOLO: return [0]
        elif stage == STAGE_DUEL: return [0]
        elif stage == STAGE_TEAM: return [0, 1]
        else: return [0, 1]

    def evolve_population(self):
        self.log(f"--- Evolution Step (Gen {self.generation}) ---")
        
        # 1. Update Metrics & EMA Calculation
        # Proficiency = AvgSteps + Penalty/(Sqrt(Hits)+1). Lower is Better.
        PENALTY_CONST = 50.0 
        
        # Collect raw values for debug logging
        debug_raw_fitness = []
        
        for p in self.population:
            # Efficiency (Avg Steps per Checkpoint)
            if p['total_cp_hits'] > 0:
                raw_avg = p['total_cp_steps'] / p['total_cp_hits']
            else:
                raw_avg = 999.0 # Penalty
            
            # Proficiency Score (Raw)
            prof_score = raw_avg + (PENALTY_CONST / (np.sqrt(p['total_cp_hits'] + 1)))
            p['efficiency_score'] = prof_score # Store raw for logs
            
            # Extract other raw metrics
            wins = p['wins']
            matches = p.get('matches', 0)
            
            # PBT Win Rate Calculation (Safe Division)
            if matches > 0:
                win_rate = float(wins) / float(matches)
            else:
                win_rate = 0.0
            
            laps = p['laps_score']
            checkpoints = p['checkpoints_score']
            
            # Calculate Behavior Vector [Speed, Steer, MateDist]
            buf = p['behavior_buffer']
            count = buf[3].item()
            if count > 0:
                avg_speed = buf[0].item() / count
                avg_steer = buf[1].item() / count
                avg_dist = buf[2].item() / count
            else:
                avg_speed = 0.0
                avg_steer = 0.0
                avg_dist = 0.0
            
            # Update Behavior Vector
            # We treat this as EMA too for smoothness
            p_beh = torch.tensor([avg_speed, avg_steer, avg_dist], dtype=torch.float32)
            p['behavior'] = p_beh # Store current gen behavior
            
            # --- Update EMAs ---
            # Helper to update
            def update_ema(current_val, old_ema, alpha):
                if old_ema is None: return current_val
                return alpha * current_val + (1.0 - alpha) * old_ema
                
            p['ema_efficiency'] = update_ema(prof_score, p['ema_efficiency'], self.ema_alpha)
            p['ema_wins'] = update_ema(win_rate, p['ema_wins'], self.ema_alpha)
            p['ema_consistency'] = update_ema(float(checkpoints), p['ema_consistency'], self.ema_alpha)
            p['ema_laps'] = update_ema(float(laps), p['ema_laps'], self.ema_alpha)
            
            p['ema_runner_vel'] = update_ema(p['avg_runner_vel'], p['ema_runner_vel'], self.ema_alpha)
            p['ema_blocker_dmg'] = update_ema(p['avg_blocker_dmg'], p['ema_blocker_dmg'], self.ema_alpha)
            
            # EMA Behavior is tricky (Vector)
            # Just overwrite for now or EMA vector? EMA vector is better.
            # p['behavior'] contains current.
            # We need p['ema_behavior']
            if p.get('ema_behavior') is None:
                p['ema_behavior'] = p_beh
            else:
                p['ema_behavior'] = self.ema_alpha * p_beh + (1.0 - self.ema_alpha) * p['ema_behavior']
            
            # Store in 'behavior' for novelty calc (using EMA behavior)
            p['behavior'] = p['ema_behavior']
            

        # 2. Calculate Diversity (Novelty)
        # Uses p['behavior'] which is now the EMA behavior
        novelty_scores = calculate_novelty(self.population, k=5)
        for i, p in enumerate(self.population):
            p['novelty_score'] = novelty_scores[i]
            p['ema_dist'] = novelty_scores[i] # Just log it
            
        # 3. Define Objectives for NSGA-II
        # We need to format objective matrix [N, M]
        # We assume HIGHER IS BETTER for all columns passed to sort.
        # So we invert Efficiency (Lower is better).
        
        stage = self.env.curriculum_stage
        
        objectives_list = []
        
        if stage == STAGE_SOLO:
            # Objectives: 
            # 1. Consistency (EMA Checkpoints) -> Max
            # 2. Efficiency (EMA Proficiency) -> Minimize (So Maximize -EMA)
            # 3. Novelty -> Max
            
            for p in self.population:
                obj = [
                    p['ema_consistency'],
                    -p['ema_efficiency'], 
                    p['novelty_score'] * 100.0
                ]
                objectives_list.append(obj)
                
        elif stage == STAGE_DUEL:
            # Objectives:
            # 1. Win Rate (EMA Win Rate) -> Max
            # 2. Efficiency -> Minimize
            # 3. Novelty -> Max
            
            for p in self.population:
                obj = [
                    p['ema_wins'],
                    -p['ema_efficiency'],
                    p['novelty_score'] * 100.0
                ]
                objectives_list.append(obj)

        elif stage == STAGE_TEAM:
            # Objectives (Role-Specific):
            # 1. Win Rate (EMA Win Rate) -> Max (Primary)
            # 2. Runner Velocity (EMA) -> Max
            # 3. Blocker Damage (EMA) -> Max
            
            for p in self.population:
                 # Safety check for Nones
                rv = p['ema_runner_vel'] if p['ema_runner_vel'] is not None else 0.0
                bd = p['ema_blocker_dmg'] if p['ema_blocker_dmg'] is not None else 0.0
                
                obj = [
                    p['ema_wins'],
                    rv,
                    bd
                ]
                objectives_list.append(obj)
                
        else: # LEAGUE
             # Objectives:
             # 1. Win Rate (EMA Win Rate) -> Max
             # 2. Laps (EMA Laps) -> Max
             # 3. Efficiency (EMA Proficiency) -> Minimize (So Maximize -EMA)
             for p in self.population:
                obj = [
                    p['ema_wins'], # Win Rate
                    p['ema_laps'], # Raw Laps
                    -p['ema_efficiency']
                ]
                objectives_list.append(obj)

        objectives_np = np.array(objectives_list)
        
        # 4. NSGA-II Sort
        fronts = fast_non_dominated_sort(objectives_np)
        
        # Calculate Crowding Distance
        crowding = calculate_crowding_distance(objectives_np, fronts)
        
        # Assign Fronts and Rank to Agents for Logging
        for rank, front in enumerate(fronts):
            for pid in front:
                self.population[pid]['rank'] = rank
                self.population[pid]['crowding'] = crowding[pid]
                
        # 5. Select Elites (Front 0)
        # Sort Front 0 by Crowding Distance Descending
        front0_indices = fronts[0]
        front0_indices.sort(key=lambda x: crowding[x], reverse=True)
        
        elites = [self.population[i] for i in front0_indices[:2]]
        # Fallback if front 0 is small (unlikely)
        if len(elites) < 2:
            # Pick from Front 1
             remaining = [self.population[i] for i in range(POP_SIZE) if i not in front0_indices]
             # Sort remaining by Rank ASC, Crowding DESC
             remaining.sort(key=lambda x: (x['rank'], -x['crowding']))
             elites.extend(remaining[: 2 - len(elites)])
        
        # Logging
        elite_ids = [p['id'] for p in elites]
        self.log(f"Pareto Fronts: {[len(f) for f in fronts]}")
        self.log(f"Elites (Rank 0, Crowded): {elite_ids}")
        self.log(f"Top Stats: Eff {elites[0]['ema_efficiency']:.1f}, Wins {elites[0]['ema_wins']:.1f}, Nov {elites[0]['novelty_score']:.2f}")

        # 6. Tournament Selection & Replacement
        # Identify Culls (Bottom 25% by Rank/Crowding)
        # Full sort of population
        # Key: (Rank ASC, Crowding DESC)
        
        # We want to keep Lower Rank (0 is best), Higher Crowding.
        sorted_pop_indices = list(range(POP_SIZE))
        sorted_pop_indices.sort(key=lambda i: (self.population[i]['rank'], -self.population[i]['crowding']))
        
        num_culls = max(2, int(POP_SIZE * 0.25))
        cull_indices = sorted_pop_indices[-num_culls:]
        parent_candidates = sorted_pop_indices[:-num_culls]
        
        # Replacement Logic
        for idx in cull_indices:
            loser = self.population[idx]
            
            # Tournament Selection from candidates
            # Pick 3, choose Best (lowest rank, highest crowding)
            competitors = random.sample(parent_candidates, 3)
            competitors.sort(key=lambda i: (self.population[i]['rank'], -self.population[i]['crowding']))
            parent_idx = competitors[0]
            parent = self.population[parent_idx]
            
            # Clone Logic
            loser['agent'].load_state_dict(parent['agent'].state_dict())
            loser['lr'] = parent['lr']
            loser['ent_coef'] = parent.get('ent_coef', ENT_COEF)
            
            # Clone EMAs! (Robustness Inheritance)
            loser['ema_efficiency'] = parent['ema_efficiency']
            loser['ema_wins'] = parent['ema_wins']
            loser['ema_consistency'] = parent['ema_consistency']
            loser['ema_laps'] = parent['ema_laps']
            loser['ema_runner_vel'] = parent['ema_runner_vel']
            loser['ema_blocker_dmg'] = parent['ema_blocker_dmg']
            loser['ema_behavior'] = parent['ema_behavior'].clone() if parent.get('ema_behavior') is not None else None
            
            # Mutate Rewards (Still useful for PPO inner loop)
            new_weights = parent['weights'].copy()
            keys = list(new_weights.keys())
            num_mutations = random.choice([1, 2])
            for _ in range(num_mutations):
                k = random.choice(keys)
                factor = random.uniform(0.7, 1.3)
                new_weights[k] *= factor
            loser['weights'] = new_weights
            
            # Mutate Hyperparams
            if random.random() < 0.3:
                loser['lr'] *= random.uniform(0.8, 1.2)
                
            # Reset Optimizer to clear bad momentum!
            # We must create a NEW optimizer instance with the parameters of the cloned model.
            loser['optimizer'] = optim.Adam(loser['agent'].parameters(), lr=loser['lr'], eps=1e-5)
            
            if random.random() < 0.3:
                loser['ent_coef'] *= random.uniform(0.8, 1.2)
                loser['ent_coef'] = max(0.0001, min(0.1, loser['ent_coef']))
                
            if random.random() < 0.3:
                loser['clip_range'] *= random.uniform(0.8, 1.2)
                loser['clip_range'] = max(0.05, min(0.4, loser['clip_range']))
                
            self.log(f"Agent {loser['id']} (Rank {loser['rank']}) replaced by clone of {parent['id']} (Rank {parent['rank']})")
             # Update Global Tensor
            start_idx = loser['id'] * ENVS_PER_AGENT
            end_idx = start_idx + ENVS_PER_AGENT
            for k, v in new_weights.items():
                self.reward_weights_tensor[start_idx:end_idx, k] = v
                 
        # 5. Save & Reset
        self.save_generation()
        
        for p in self.population:
            # Reset Current Gen Counters
            p['laps_score'] = 0
            p['checkpoints_score'] = 0
            p['reward_score'] = 0.0
            p['wins'] = 0
            p['matches'] = 0
            p['max_streak'] = 0
            p['total_cp_steps'] = 0
            p['total_cp_hits'] = 0
            p['avg_runner_vel'] = 0.0
            p['avg_blocker_dmg'] = 0.0
            # Reset Behavior Buffer
            p['behavior_buffer'].zero_()
            
        self.generation += 1
        # --- LEADER SELECTION (Performance Based) ---
        # Select strictly best performer from Front 0 (or population if empty)
        candidates = fronts[0] 
        if not candidates: candidates = range(len(self.population))
        
        if self.env.curriculum_stage == STAGE_SOLO:
             # Combined Metric: Consistency - Efficiency (Maximize)
             # Higher Consistency is better (Progress)
             # Lower Efficiency is better (Speed)
             # Maximize (Cons - Eff)
             def combined_score(idx):
                 p = self.population[idx]
                 eff = p.get('ema_efficiency', 999.0) if p.get('ema_efficiency') is not None else 999.0
                 cons = p.get('ema_consistency', 0.0) if p.get('ema_consistency') is not None else 0.0
                 return cons - eff
                 
             best_guy = max(candidates, key=combined_score)
        else:
             # Max Win Rate (Best)
             best_guy = max(candidates, key=lambda i: self.population[i].get('ema_wins', 0.0) if self.population[i].get('ema_wins') is not None else 0.0)
             
        self.leader_idx = self.population[best_guy]['id']

    def save_generation(self):
        gen_dir = f"data/generations/gen_{self.generation}"
        os.makedirs(gen_dir, exist_ok=True)
        self.log(f"Saving generation {self.generation} to {gen_dir}...")
        
        # Identify Top 2 for League (Consistent with evolve logic)
        sorted_pop = sorted(self.population, key=lambda x: (x['laps_score'], x['checkpoints_score'], x['reward_score']), reverse=True)
        top_2 = sorted_pop[:2]
        
        for p in self.population:
            agent_id = p['id']
            save_path = os.path.join(gen_dir, f"agent_{agent_id}.pt")
            torch.save(p['agent'].state_dict(), save_path)
            
            # Register Top Agents to League
            # START SOTA UPDATE: Enforce strict entry criteria
            # Only save to League if passing the "Racer" bar (Stage 1+ and Diff > 0.5)
            if self.env.curriculum_stage >= STAGE_DUEL and self.env.bot_difficulty > 0.5:
                if p in top_2:
                    league_name = f"gen_{self.generation}_agent_{agent_id}"
                    metrics = {
                        "efficiency": p.get('ema_efficiency', 999.0),
                        "consistency": p.get('ema_consistency', 0.0),
                        "wins_ema": p.get('ema_wins', 0.0),
                        "novelty": p.get('novelty_score', 0.0)
                    }
                    self.league.register_agent(league_name, save_path, self.generation, metrics=metrics)
            # END SOTA UPDATE
                
        # Save Normalization Stats (Global)
        rms_path = os.path.join(gen_dir, "rms_stats.pt")
        torch.save({
            'self': self.rms_self.state_dict(),
            'ent': self.rms_ent.state_dict(),
            'cp': self.rms_cp.state_dict()
        }, rms_path)
    def log_iteration_summary(self, global_step, sps, current_tau, avg_loss):
        leader = self.population[self.leader_idx]
        
        # Calculate Pop Stats
        effs = [p.get('ema_efficiency', 999.0) for p in self.population if p.get('ema_efficiency') is not None]
        cons = [p.get('ema_consistency', 0.0) for p in self.population if p.get('ema_consistency') is not None]
        wins = [p.get('ema_wins', 0.0) for p in self.population if p.get('ema_wins') is not None]
        novs = [p.get('novelty_score', 0.0) for p in self.population]
        
        avg_eff = np.mean(effs) if effs else 999.0
        avg_con = np.mean(cons) if cons else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_nov = np.mean(novs) if novs else 0.0
        
        # Leader Stats
        l_eff = leader.get('ema_efficiency')
        if l_eff is None: l_eff = 999.0
        
        l_con = leader.get('ema_consistency')
        if l_con is None: l_con = 0.0
        
        l_win = leader.get('ema_wins')
        if l_win is None: l_win = 0.0
        
        l_nov = leader.get('novelty_score')
        if l_nov is None: l_nov = 0.0
        
        # Format Table
        border = "=" * 80
        
        # Get Current Step Penalty (Mean, as it might vary slightly or be uniform)
        curr_step_pen = self.reward_weights_tensor[:, RW_STEP_PENALTY].mean().item()
        
        self.log(border)
        self.log(f" ITERATION {self.iteration} | Gen {self.generation} | Step {global_step} | SPS {sps}")
        self.log(f" Stage: {self.env.curriculum_stage} | Difficulty: {self.env.bot_difficulty:.2f} | Tau: {current_tau:.2f} | Step Pen: {curr_step_pen:.1f}")
        self.log("-" * 80)
        self.log(f" {'Metric':<15} | {'Leader':<10} | {'Pop Avg':<10} | {'Best':<10}")
        self.log("-" * 80)
        self.log(f" {'Efficiency':<15} | {l_eff:<10.1f} | {avg_eff:<10.1f} | {min(effs) if effs else 0:<10.1f}")
        self.log(f" {'Consistency':<15} | {l_con:<10.1f} | {avg_con:<10.1f} | {max(cons) if cons else 0:<10.1f}")
        self.log(f" {'Wins (EMA)':<15} | {l_win:<10.1f} | {avg_win:<10.1f} | {max(wins) if wins else 0:<10.1f}")
        self.log(f" {'Novelty':<15} | {l_nov:<10.2f} | {avg_nov:<10.2f} | {max(novs) if novs else 0:<10.2f}")
        self.log("-" * 80)
        self.log(f" Loss: {avg_loss:.4f}")
        self.log(border)

    def update_step_penalty_annealing(self):
        """
        Anneals Step Penalty based on Curriculum Stage and Bot Difficulty.
        Stage 0 (Solo): Full Penalty (10.0)
        Stage > 0: Linear Decay based on Bot Difficulty.
        """
        stage = self.env.curriculum_stage
        base_penalty = DEFAULT_REWARD_WEIGHTS[RW_STEP_PENALTY] # 10.0
        
        if stage == STAGE_LEAGUE:
            # League Mode: No Step Penalty (Pure Win/Loss/Metrics)
            new_val = 0.0
        elif stage == STAGE_SOLO:
            # Full Penalty
            new_val = base_penalty
        else:
            # Anneal: Val = Base * (1.0 - Diff * 0.8)
            # At Diff 0.0 -> 10.0
            # At Diff 1.0 -> 2.0
            anneal_factor = 1.0 - (self.env.bot_difficulty * 0.8)
            new_val = base_penalty * anneal_factor
            
        # Update Tensor
        self.reward_weights_tensor[:, RW_STEP_PENALTY] = new_val
        
    def train_loop(self, stop_event=None, telemetry_callback=None):
        self.log(f"Starting Evolutionary PPO (Pop: {POP_SIZE}, Envs/Agent: {ENVS_PER_AGENT})...")
        self.running = True
        
        # Save Initial Generation (Gen 0)
        self.save_generation()
        
        # Explicit Reset to apply initial curriculum stage (e.g. Stage 1/2 from Custom Launch)
        self.env.reset()
        
        global_step = 0
        
        obs_data = self.env.get_obs()
        start_time = time.time()
        sps = 0 # Initialize to avoid unbound error on first steps
        
        # Helper stacker REMOVED - using direct tensor slicing

        while global_step < TOTAL_TIMESTEPS:
            if stop_event and stop_event.is_set(): break

            # Check Curriculum & Update Config BEFORE starting the iteration
            prev_stage = self.env.curriculum_stage
            self.check_curriculum()
            
            # If Stage Changed, we MUST reset the environment to spawn new pods/apply new logic immediately.
            if self.env.curriculum_stage != prev_stage:
                 self.log(f"Config: Stage Transition {prev_stage} -> {self.env.curriculum_stage}. Resetting Environment...")
                 
                 # --- Mitosis Strategy (Transition to Team Mode) ---
                 # If moving from Solo/Duel (Stage < 2) to Team (Stage 2),
                 # Clone Runner Brain to Blocker Brain to avoid "Dead Weight".
                 if prev_stage < STAGE_TEAM and self.env.curriculum_stage >= STAGE_TEAM:
                      self.log("Approaching Team Stage: Executing Mitosis (Cloning Runner -> Blocker)...")
                      for p in self.population:
                           agent = p['agent']
                           # Clone Weights
                           agent.blocker_actor.load_state_dict(agent.runner_actor.state_dict())
                           # Reset Optimizer (Critical to break old momentum)
                           p['optimizer'] = optim.Adam(agent.parameters(), lr=p['lr'], eps=1e-5)
                           
                 self.env.reset()
                 obs_data = self.env.get_obs()


            # --- Dynamic Config Check ---
            current_stage = self.env.curriculum_stage
            target_steps = 256
            target_evolve = 8
            
            if current_stage == STAGE_SOLO:
                target_steps = 512
                target_evolve = 4 # Frequent updates for micro-opt
            elif current_stage == STAGE_DUEL:
                target_steps = 512
                target_evolve = 4
            elif current_stage == STAGE_TEAM:
                target_steps = 256
                # Dynamic Interval: Diff 0.0 -> 16, Diff 1.0 -> 4
                target_evolve = int(16 - 12 * self.env.bot_difficulty)
                target_evolve = max(4, target_evolve) # Safety clamp
            elif current_stage == STAGE_LEAGUE:
                target_steps = 512
                target_evolve = 8 # Stable league training
            
            # Apply Evolve Interval
            self.current_evolve_interval = target_evolve         
            # Check Active Pods Change
            current_active_pods_ids = self.get_active_pods()
            active_count = len(current_active_pods_ids)
            
            needs_realloc = False
            
            if target_steps != self.current_num_steps:
                 self.log(f"Stage Change Triggered Config Update: Steps {self.current_num_steps} -> {target_steps}")
                 self.current_num_steps = target_steps
                 needs_realloc = True
                 
            if active_count != self.current_active_pods_count:
                 self.log(f"Stage Change Triggered Config Update: Active Pods {self.current_active_pods_count} -> {active_count}")
                 needs_realloc = True

            # DEBUG LOG
            if current_stage == STAGE_TEAM and self.current_active_pods_count == 1:
                self.log(f"DEBUG: Stage IS Team, but Alloc Count is 1. Active Count: {active_count}. Needs Realloc: {needs_realloc}")

            if needs_realloc:
                 self.log("DEBUG: Calling allocate_buffers()...")
                 self.allocate_buffers() # Re-allocate
                 self.log(f"DEBUG: New Batch Shape: {self.agent_batches[0]['self_obs'].shape}")
                 
            if target_evolve != self.current_evolve_interval:
                 self.log(f"Config Update: Evolve Interval {self.current_evolve_interval} -> {target_evolve}")
                 self.current_evolve_interval = target_evolve

            # Start Iteration
            self.iteration += 1
            if self.iteration % 1 == 0:
                self.log(f"Starting iteration {self.iteration}")
            
            # --- PRE-ITERATION LEAGUE LOGIC ---
            # Sample Opponent for this iteration (Fictitious Self-Play)
            self.opponent_agent_loaded = False
            # Only sample if we are in League Mode
            if self.env.curriculum_stage >= STAGE_LEAGUE:
                # SOTA: PFSP - Prioritize based on Leader's historical performance
                leader_id_str = f"gen_{self.generation}_agent_{self.population[self.leader_idx]['id']}"
                opp_path = self.league.sample_opponent(active_agent_id=leader_id_str, mode='pfsp')
                if opp_path and os.path.exists(opp_path):
                    try:
                        # Load Main Opponent (for Main Population)
                        self.opponent_agent.load_state_dict(torch.load(opp_path, map_location=self.device))
                        self.opponent_agent.eval()
                        self.opponent_agent_loaded = True
                        
                        self.current_opponent_id = os.path.basename(opp_path).replace(".pt", "")
                        self.log(f"⚔️  LEAGUE MATCH: Main Pop vs {self.current_opponent_id} ⚔️")
                        
                        # Snapshot for PFSP Update
                        self.iteration_start_wins = self.population[self.leader_idx]['wins']
                        self.iteration_start_matches = self.population[self.leader_idx]['matches']
                        
                    except Exception as e:
                        self.log(f"Failed to load opponent {opp_path}: {e}")
                        self.opponent_agent_loaded = False
                        
                    # --- NEW: Exploiter Opponent Logic ---
                    # Exploiters play against CURRENT Leader (or recent strong agent)
                    # We can use 'latest' mode from LeagueManager if implemented, or just pick the Leader of Gen N-1.
                    # Since we are in Gen N, Gen N-1 is in the League Registry.
                    
                    # Check if we have exploiters
                    has_exploiters = any(p['type'] == 'exploiter' for p in self.population)
                    self.exploiter_opponent_loaded = False
                    
                    if has_exploiters:
                         # Use 'latest' to get the strongest recent agent
                         exp_opp_path = self.league.sample_opponent(active_agent_id=leader_id_str, mode='latest')
                         if exp_opp_path:
                             try:
                                 self.exploiter_opponent_agent.load_state_dict(torch.load(exp_opp_path, map_location=self.device))
                                 self.exploiter_opponent_agent.eval()
                                 self.exploiter_opponent_loaded = True
                                 exp_id = os.path.basename(exp_opp_path).replace(".pt", "")
                                 self.log(f"🕵️  EXPLOITER MATCH: Exploiters vs {exp_id} (Targeting Leader)")
                             except Exception as e:
                                 self.log(f"Failed to load exploiter opponent {exp_opp_path}: {e}")
                else:
                    # Fallback to Mirror Self-Play
                    pass
            active_pods = self.get_active_pods()
            num_active_per_agent = len(active_pods) * ENVS_PER_AGENT
            
            # --- Collection Phase (Parallel) ---
            # Batches for each agent
            # Using self.agent_batches (allocated dynamically)
                 
            # Unpack Obs
            # obs_data is tuple (self, tm, en, cp) tensors [4, NUM_ENVS, ...]
            all_self, all_tm, all_en, all_cp = obs_data
            
            for step in range(NUM_STEPS):
                # --- Global Normalization ---
                # Normalize ALL active observations at once for efficiency
                # Source: [4, NUM_ENVS, ...]
                
                # Active Pods Slicing
                # active_pods is list [0] or [0, 1] etc.
                # all_self[active_pods] returns [N_Active, NUM_ENVS, 14]
                # It copies and stacks them. Since all_self[i] is [NUM_ENVS, 14] contiguous, this is fast.
                
                raw_self = all_self[active_pods].view(-1, 14) # [N_Active * NUM_ENVS, 14]
                raw_tm = all_tm[active_pods].view(-1, 13)
                raw_en = all_en[active_pods].view(-1, 2, 13)
                raw_cp = all_cp[active_pods].view(-1, 6)
                
                # Normalize & Update Stats
                norm_self = self.rms_self(raw_self)
                
                # Normalize Teammate (13) using ENT stats
                norm_tm = self.rms_ent(raw_tm)
                
                # Normalize Enemies (2, 13) using ENT stats
                # Flatten -> Norm -> Reshape
                total_rows_en, N_en, D_ent = raw_en.shape
                norm_en = self.rms_ent(raw_en.view(-1, D_ent)).view(total_rows_en, N_en, D_ent)
                
                norm_cp = self.rms_cp(raw_cp)
                
                # 1. Inference per Agent
                full_actions = []
                
                # The normalized tensors are flat [N_Active * NUM_ENVS, ...]
                # Order: Pod 0 Envs... Pod 1 Envs...
                # BUT wait.
                # all_self[active_pods] behavior:
                # If active_pods = [0, 1, 2, 3]
                # Result is [4, 4096, 14]
                # .view(-1, 14) -> Pod 0 (all envs), Pod 1 (all envs)...
                
                # Batch Slicing Needs to match this.
                # Agent i owns envs start:end
                # We need [Pod 0 start:end, Pod 1 start:end...]
                # This is tricky if flattened this way.
                
                # Actually, simpler:
                # Keep normalized as [N_Active, NUM_ENVS, D]
                
                norm_self = norm_self.view(len(active_pods), NUM_ENVS, 14)
                norm_tm = norm_tm.view(len(active_pods), NUM_ENVS, 13)
                norm_en = norm_en.view(len(active_pods), NUM_ENVS, 2, 13)
                norm_cp = norm_cp.view(len(active_pods), NUM_ENVS, 6)
                
                # Now Agent i needs envs i*128 : (i+1)*128
                # across ALL active pods.
                # So we slice dim 1.
                
                for i in range(POP_SIZE):
                    start_env = i * ENVS_PER_AGENT
                    end_env = start_env + ENVS_PER_AGENT
                    
                    # Slice Normalized Tensors [N_Active, Batch, D]
                    # We want [N_Active * Batch, D] for input to network
                    # But careful with ordering. DeepSets treats input as Batch independent.
                    # Flattening [N_A, B, D] -> [N_A*B, D] stacks pods: Pod0(Batch), Pod1(Batch).
                    # This is fine.
                    
                    t0_self = norm_self[:, start_env:end_env, :].reshape(-1, 14)
                    t0_tm = norm_tm[:, start_env:end_env, :].reshape(-1, 13)
                    t0_en = norm_en[:, start_env:end_env, :, :].reshape(-1, 2, 13)
                    t0_cp = norm_cp[:, start_env:end_env, :].reshape(-1, 6)
                    
                    with torch.no_grad():
                        agent = self.population[i]['agent']
                        action0, logprob0, _, value0 = agent.get_action_and_value(t0_self, t0_tm, t0_en, t0_cp)
                        
                    # Store
                    # Agent batch storage expectation:
                    # 'self_obs': [NUM_STEPS, n_active_per_agent, 14]
                    # n_active_per_agent = N_Active * ENVS_PER_AGENT
                    # t0_self matches this size.
                    
                    
                    batch = self.agent_batches[i]
                    batch['self_obs'][step] = t0_self.detach()
                    batch['teammate_obs'][step] = t0_tm.detach()
                    batch['enemy_obs'][step] = t0_en.detach()
                    batch['cp_obs'][step] = t0_cp.detach()
                    batch['actions'][step] = action0.detach()
                    batch['logprobs'][step] = logprob0.detach()
                    batch['values'][step] = value0.flatten().detach()
                    
                    full_actions.append(action0)

                # 2. Step Environment (Global)
                # We need to construct [4096, 4, 4] action tensor
                # full_actions is list of [N_Active * 512, 4]
                # We need to map back to env structure
                
                # Simple logic:
                if len(active_pods) == 1:
                     combined_act0 = torch.cat(full_actions, dim=0) # [4096, 4]
                     act_map = { active_pods[0]: combined_act0 }
                else:
                     combined_act0 = torch.cat(full_actions, dim=0) # [4096 * 2, 4] (stacked)
                     chunks = torch.chunk(combined_act0, len(active_pods), dim=0)
                     act_map = { pid: c for pid, c in zip(active_pods, chunks) }
                
                # Check for League Mode Opponent Logic (Stage 2: 2v2)
                if self.env.curriculum_stage >= STAGE_LEAGUE:
                     # We need to control Pods [2, 3] (Opponent Team).
                     # Strategy:
                     # 1. Use League Opponent if available.
                     # 2. Else: Use Self-Play (Current Agent).

                     use_league_opp = hasattr(self, 'opponent_agent_loaded') and self.opponent_agent_loaded
                     
                     # Opponent controls Pods 2 and 3
                     opp_indices = [2, 3] 
                     
                     # Extract Raw Obs for Opponent from GLOBAL obs tensors (all_self has 4 pods)
                     # Keep dim 1 (Envs) intact for slicing: [2, 4096, 14]
                     opp_raw_self = all_self[opp_indices] 
                     opp_raw_tm = all_tm[opp_indices]
                     opp_raw_en = all_en[opp_indices]
                     opp_raw_cp = all_cp[opp_indices]
                     
                     # Normalize (Fixed stats)
                     # Must flatten to normalize then reshape back
                     B_o, N_e, D_o = opp_raw_self.shape
                     opp_norm_self = self.rms_self(opp_raw_self.view(-1, D_o), fixed=True).view(B_o, N_e, D_o)
                     
                     D_tm = opp_raw_tm.shape[-1]
                     opp_norm_tm = self.rms_ent(opp_raw_tm.view(-1, D_tm), fixed=True).view(B_o, N_e, D_tm)
                     
                     # Enemies: [2, 4096, 2, 13]
                     D_en = opp_raw_en.shape[-1]
                     opp_norm_en = self.rms_ent(opp_raw_en.view(-1, D_en), fixed=True).view(B_o, N_e, 2, D_en)
                     
                     D_cp = opp_raw_cp.shape[-1]
                     opp_norm_cp = self.rms_cp(opp_raw_cp.view(-1, D_cp), fixed=True).view(B_o, N_e, D_cp)
                     
                     # Split Indices
                     # Main: Agents 0-27 -> Envs 0 to 28*128 = 3584
                     # Exploiter: Agents 28-31 -> Envs 3584 to 4096
                     split_idx = 28 * ENVS_PER_AGENT
                     
                     # --- MAIN GROUP INFERENCE ---
                     # Slice: Everything before split_idx
                     # Reshape to [Batch, D] for network
                     m_self = opp_norm_self[:, :split_idx, :].reshape(-1, D_o)
                     m_tm = opp_norm_tm[:, :split_idx, :].reshape(-1, D_tm)
                     m_en = opp_norm_en[:, :split_idx, :, :].reshape(-1, 2, D_en)
                     m_cp = opp_norm_cp[:, :split_idx, :].reshape(-1, D_cp)
                     
                     with torch.no_grad():
                         if use_league_opp:
                             # Main plays against "opponent_agent" (Standard History)
                             act_m, _, _, _ = self.opponent_agent.get_action_and_value(m_self, m_tm, m_en, m_cp)
                         else:
                             act_m, _, _, _ = self.agent.get_action_and_value(m_self, m_tm, m_en, m_cp)

                     # Reshape back to [2, M_Envs, 4]
                     act_m = act_m.view(2, split_idx, 4)

                     # --- EXPLOITER GROUP INFERENCE ---
                     # Slice: Everything after split_idx
                     e_self = opp_norm_self[:, split_idx:, :].reshape(-1, D_o)
                     e_tm = opp_norm_tm[:, split_idx:, :].reshape(-1, D_tm)
                     e_en = opp_norm_en[:, split_idx:, :, :].reshape(-1, 2, D_en)
                     e_cp = opp_norm_cp[:, split_idx:, :].reshape(-1, D_cp)
                     
                     with torch.no_grad():
                         # Exploiters play against "exploiter_opponent_agent" (Leader) if loaded
                         if hasattr(self, 'exploiter_opponent_loaded') and self.exploiter_opponent_loaded:
                             act_e, _, _, _ = self.exploiter_opponent_agent.get_action_and_value(e_self, e_tm, e_en, e_cp)
                         elif use_league_opp:
                             # Fallback to Main Opponent if specialized one failed to load
                             act_e, _, _, _ = self.opponent_agent.get_action_and_value(e_self, e_tm, e_en, e_cp)
                         else:
                             # Fallback to Self
                             act_e, _, _, _ = self.agent.get_action_and_value(e_self, e_tm, e_en, e_cp)

                     # Reshape back to [2, E_Envs, 4]
                     # Note: E_Envs = Total - split_idx
                     act_e = act_e.view(2, NUM_ENVS - split_idx, 4)
                     
                     # --- COMBINE ---
                     # Concatenate along Env dim (dim 1)
                     combined_opp_act = torch.cat([act_m, act_e], dim=1) # [2, 4096, 4]
                     
                     # Map to Act Map
                     act_map[2] = combined_opp_act[0] # Pod 2
                     act_map[3] = combined_opp_act[1] # Pod 3
                
                # Construct Global Action Tensor
                env_actions = torch.zeros((NUM_ENVS, 4, 4), device=self.device)
                
                for pid, act in act_map.items():
                    env_actions[:, pid] = act

                
                # Calculate Tau (Dense Reward Annealing)
                # 0.0 (Full Dense) -> 1.0 (Full Sparse/Shaped)
                # Stage 0: 0.0
                # Stage 1: 0.5
                # Stage 2: 0.9
                current_tau = 0.0
                if self.env.curriculum_stage == STAGE_DUEL:
                    current_tau = 0.5
                elif self.env.curriculum_stage == STAGE_TEAM:
                    current_tau = 0.0
                elif self.env.curriculum_stage == STAGE_LEAGUE:
                    current_tau = 0.9
                
                # STEP
                # Pass Global Reward Tensor
                # info is dictionary now
                # rewards is [NUM_ENVS, 2] (Team 0, Team 1)
                # dones is [NUM_ENVS]
                rewards_all, dones, infos = self.env.step(env_actions, reward_weights=self.reward_weights_tensor, tau=current_tau, team_spirit=self.team_spirit)
                
                # --- Reward Processing ---
                # Blending is now handled in Env (Hybrid: Individual Dense + Shared Sparse)
                blended_rewards = rewards_all
                
                # --- Reward Normalization ---
                # 1. Update Returns Buffer (All 4 channels)
                dones_exp = dones.unsqueeze(1).expand(-1, 4)
                self.returns_buffer = blended_rewards + GAMMA * self.returns_buffer * (~dones_exp).float()
                
                # 2. Update RMS (Flattened)
                self.rms_ret.update(self.returns_buffer.view(-1))
                
                # 3. Normalize
                rew_var = torch.clamp(self.rms_ret.var, min=1e-4)
                rew_std = torch.sqrt(rew_var)
                norm_rewards = torch.clamp(blended_rewards / rew_std, -10.0, 10.0) # [B, 4]
                
                # --- Intrinsic Curiosity Update ---
                # RND Logic
                rnd_input = norm_self.reshape(-1, 14).detach() # [N_Active * NUM_ENVS, 14]
                intrinsic_rewards = self.rnd.compute_intrinsic_reward(rnd_input) # [N_Active * NUM_ENVS]
                
                # Map Intrinsic to Pods
                r_int = intrinsic_rewards.view(len(active_pods), NUM_ENVS)
                
                if self.env.curriculum_stage == STAGE_SOLO:
                     # Only add to Pod 0 (active_pods[0])
                     norm_rewards[:, 0] += r_int[0] * self.rnd_coef

                # Split rewards back to agents
                for i in range(POP_SIZE):
                    start_env = i * ENVS_PER_AGENT
                    end_env = start_env + ENVS_PER_AGENT
                    
                    r_chunks = []
                    d_chunks = []
                    
                    raw_sum = 0
                    raw_count = 0
                    
                    d_slice = dones[start_env:end_env]
                    
                    for p_idx, pid in enumerate(active_pods):
                        r_slice = norm_rewards[start_env:end_env, pid]
                        r_chunks.append(r_slice)
                        d_chunks.append(d_slice)
                        
                        # Logging Raw (Blended or Pure? Let's log Blended as that's what we see)
                        # Actually pure env feedback is useful (rewards_all).
                        raw_r = rewards_all[start_env:end_env, pid]
                        raw_sum += raw_r.mean().item()
                        raw_count += 1
                        
                    flat_r = torch.cat(r_chunks, dim=0)
                    flat_d = torch.cat(d_chunks, dim=0)
                        
                    self.agent_batches[i]['rewards'][step] = flat_r.detach()
                    self.agent_batches[i]['dones'][step] = flat_d.float().detach()
                    
                    # Track Score for Evolution (Use RAW rewards)
                    if raw_count > 0:
                        self.population[i]['reward_score'] += (raw_sum / raw_count)
                    
                    # Track Cumulative Metrics (Laps & Checkpoints)
                    # infos['laps_completed'] is [4096, 4]
                    # We only care about active pods for this agent
                    # Since evolution is based on "Agent Performance", we sum ALL events by pods controlled by this agent.
                    
                    # Checkpoints
                    # active_pods are [0], [0,2] or [0,1,2,3]
                    # Filter infos first
                    agent_infos_laps = infos['laps_completed'][start_env:end_env] # [128, 4]
                    agent_infos_cps = infos['checkpoints_passed'][start_env:end_env] # [128, 4]
                    
                    # Mask by active pods
                    active_laps = 0
                    active_cps = 0
                    for pid in active_pods:
                         active_laps += agent_infos_laps[:, pid].sum().item()
                         active_cps += agent_infos_cps[:, pid].sum().item()
                         
                    self.population[i]['laps_score'] += active_laps
                    self.population[i]['checkpoints_score'] += active_cps
                    
                    # Accumulate Role Metrics
                    # Velocity is "Speed per Step", so we sum it and later divide by total_steps? 
                    # Actually logic: We likely want "Avg Velocity". 
                    # But for now, just Sum it. We can normalize later.
                    # Or better: Accumulate Sum and Count? Count = Steps * ActivePods (Already known)
                    
                    agent_infos_vel = infos['runner_velocity'][start_env:end_env] # [128, 4]
                    agent_infos_dmg = infos['blocker_damage'][start_env:end_env] # [128, 4]
                    
                    active_vel_sum = 0.0
                    active_dmg_sum = 0.0
                    
                    for pid in active_pods:
                         active_vel_sum += agent_infos_vel[:, pid].sum().item()
                         active_dmg_sum += agent_infos_dmg[:, pid].sum().item()
                         
                    self.population[i]['avg_runner_vel'] += active_vel_sum
                    self.population[i]['avg_blocker_dmg'] += active_dmg_sum
                    
                    # Track New Metrics
                    start_streak = infos['current_streak'][start_env:end_env] # [128, 4]
                    start_steps = infos['cp_steps'][start_env:end_env] # [128, 4]
                    
                    # Max Streak
                    current_max = start_streak.max().item()
                    if current_max > self.population[i]['max_streak']:
                        self.population[i]['max_streak'] = current_max
                        
                    # Efficiency
                    mask = (start_steps > 0) & (start_streak > 0)
                    if mask.any():
                        self.population[i]['total_cp_steps'] += start_steps[mask].sum().item()
                        self.population[i]['total_cp_hits'] += mask.sum().item()
                    
                    # Wins (Track from env.winners on Reset)
                    # winners is [4096] containing winner team index (0 or 1) or -1.
                    # We need to attribute this to the agent.
                    # This happens only when Done is True.
                    # The env.winners is updated in env.step? 
                    # Actually env.winners is usually set when 'dones' is set. 
                    # Taking a look at logic, we might need to rely on the fact that 
                    # if done[e] is true, we check winners[e].
                    
                    # Filter for done envs in this agent's batch
                    agent_dones = d_slice.bool()
                    if agent_dones.any():
                        # Global indices
                        # start + relative indices of dones
                        done_indices_rel = torch.nonzero(agent_dones).flatten()
                        done_indices_global = start_env + done_indices_rel
                        
                        current_winners = self.env.winners[done_indices_global] # [N_Done]
                        
                        # My pods:
                        # If Solo: Pod 0 is Team 0.
                        # If Duel: Pod 0 is Team 0.
                        # If I am Team 0 (which I am, as Agent control Pod 0/1)
                        # We just check if winner == 0.
                        
                        # Note: self.env.winners returns TEAM index (0 or 1).
                        # In Solo/Duel, 'Agent' controls Team 0.
                        # So a win is if winner == 0.
                        
                        win_count = (current_winners == 0).sum().item()
                        self.population[i]['wins'] += win_count
                        
                        # Count matches (any done is a match end)
                        match_count = agent_dones.sum().item()
                        self.population[i]['matches'] += match_count
                    
                # Telemetry (Capture BEFORE Reset to see Finish Line state)
                if telemetry_callback:
                    # Switch to Step-Based Sampling (Stride = 4 -> 25% of steps)
                    # Stride 4 + Frontend 20fps = 80 sim steps/sec rendered.
                    # Sim Physics ~60Hz -> ~1.3x Playback Speed (Smooth Real-time-ish).
                    TELEMETRY_STRIDE = 4
                    current_total_step = global_step + step
                    
                    if True: # Logic moved inside loop
                        
                        # Extract flags for batch
                        # info['collision_flags'] is [Batch, 4]
                        batch_coll_flags = infos.get("collision_flags", None)

                        for t_idx, t_env in enumerate(self.telemetry_env_indices):
                             # CRITICAL FIX: Capture if Stride Match OR Done
                             # If we miss a Done signal, the backend buffer never flushing.
                             is_stride = (current_total_step % TELEMETRY_STRIDE == 0)
                             is_done_env = dones[t_env].item()
                             
                             if is_stride or is_done_env:
                                 coll_f = None
                                 if batch_coll_flags is not None:
                                     coll_f = batch_coll_flags[t_env].cpu().numpy()

                                 telemetry_callback(global_step + step, sps, 0, 0, self.current_win_rate, t_env, 0, None, is_done_env, 
                                                    rewards_all[t_env].cpu().numpy(), env_actions[t_env].cpu().numpy(), 
                                                    collision_flags=coll_f)
                                 
                                 # Update stream target if current one finished (for NEXT step)
                                 if is_done_env:
                                     done_indices = torch.nonzero(dones).flatten()
                                     if len(done_indices) > 0:
                                         candidates = done_indices.tolist()
                                         new_idx = random.choice(candidates)
                                         self.telemetry_env_indices[t_idx] = new_idx
                
                # --- Behavior Characterization Tracking ---
                # Track Avg Speed and Steering Variance per Agent
                # self.env.physics.vel [N, 4, 2]
                # self.env.physics.angle [N, 4]
                
                # We need to map back to agents.
                # Speed scalar
                vel_active = self.env.physics.vel[:, active_pods] # [N, Active, 2]
                speed_active = torch.norm(vel_active, dim=2) # [N, Active]
                
                # Steering Action (proxy for behavior style)
                # We can track actual angle change or input action. Input action is cleaner.
                # env_actions [N, 4, 4]. 
                # Angle action is index 1.
                steer_active = torch.abs(env_actions[:, active_pods, 1]) # [N, Active]
                
                # Flatten to Env -> Agg per agent
                # Sum over Active pods
                sum_speed = speed_active.sum(dim=1) # [N]
                sum_steer = steer_active.sum(dim=1) # [N]
                
                # Teammate Distance (Only if > 1 active pod)
                # Assumes Pod 0 and 1 are teammates
                if len(active_pods) >= 2:
                    p0 = self.env.physics.pos[:, active_pods[0], :]
                    p1 = self.env.physics.pos[:, active_pods[1], :]
                    dist_vec = torch.norm(p0 - p1, dim=1) # [N]
                    # This distance applies to "The Agent".
                    # We can sum it to the count later.
                    # Since we count "Env Steps * Active Pods", adding dist once per env step is tricky scaling.
                    # Let's add dist to the buffer.
                    # Normalization: Avg Dist per Env Step.
                    sum_dist = dist_vec
                else:
                    sum_dist = torch.zeros_like(sum_speed)
                
                for i in range(POP_SIZE):
                    start_env = i * ENVS_PER_AGENT
                    end_env = start_env + ENVS_PER_AGENT
                    
                    # Sum for this agent across its envs
                    agent_speed = sum_speed[start_env:end_env].sum()
                    agent_steer = sum_steer[start_env:end_env].sum()
                    agent_dist = sum_dist[start_env:end_env].sum()
                    # Count relates to "Pod Steps" or "Env Steps"?
                    # Previous logic: count = ENVS * len(active_pods).
                    # This implies we average per Pod Step.
                    # But Dist is per "Team Step" (defined by Env Step).
                    # If we divide agent_dist by (Envs * 2), we get Avg Dist / 2.
                    # We should probably normalize properly.
                    # Let's accumulate, and later divide by "Env Steps" for Dist?
                    # Or just divide by same count and accept factor of 1/N.
                    # As long as it's consistent across agents, it's fine for Novelty.
                    
                    count = ENVS_PER_AGENT * len(active_pods)
                    
                    self.population[i]['behavior_buffer'][0] += agent_speed
                    self.population[i]['behavior_buffer'][1] += agent_steer
                    self.population[i]['behavior_buffer'][2] += agent_dist
                    self.population[i]['behavior_buffer'][3] += count

                # Manual Reset
                if dones.any():
                    reset_ids = torch.nonzero(dones).squeeze(-1)
                    self.env.reset(reset_ids)

                # Next Obs (New Start State)
                obs_data = self.env.get_obs()
                all_self, all_tm, all_en, all_cp = obs_data

            # 3. Update Phase (Per Agent)
            # Bootstrapping & Training
            total_loss = 0
            
            # Global Normalize Next Obs (Fixed=True)
            # Source: [4, NUM_ENVS, ...]
            
            # Global Normalize Next Obs (Fixed=True)
            # Source: [4, NUM_ENVS, ...]
            
            next_raw_self = all_self[active_pods].view(-1, 14)
            next_raw_tm = all_tm[active_pods].view(-1, 13)
            next_raw_en = all_en[active_pods].view(-1, 2, 13)
            next_raw_cp = all_cp[active_pods].view(-1, 6)
            
            next_norm_self = self.rms_self(next_raw_self, fixed=True)
            
            next_norm_tm = self.rms_ent(next_raw_tm, fixed=True)
            B_e, N_e, D_e = next_raw_en.shape
            next_norm_en = self.rms_ent(next_raw_en.view(-1, D_e), fixed=True).view(B_e, N_e, D_e)
            
            next_norm_cp = self.rms_cp(next_raw_cp, fixed=True)
            
            # View as [N_Act, NUM_ENVS, ...]
            next_norm_self = next_norm_self.view(len(active_pods), NUM_ENVS, 14)
            next_norm_tm = next_norm_tm.view(len(active_pods), NUM_ENVS, 13)
            next_norm_en = next_norm_en.view(len(active_pods), NUM_ENVS, 2, 13)
            next_norm_cp = next_norm_cp.view(len(active_pods), NUM_ENVS, 6)
            
            for i in range(POP_SIZE):
                batch = self.agent_batches[i]
                agent = self.population[i]['agent']
                optimizer = self.population[i]['optimizer']
                
                # Slice Next Val
                start_env = i * ENVS_PER_AGENT
                end_env = start_env + ENVS_PER_AGENT
                
                t0_self = next_norm_self[:, start_env:end_env, :].reshape(-1, 14)
                t0_tm = next_norm_tm[:, start_env:end_env, :].reshape(-1, 13)
                t0_en = next_norm_en[:, start_env:end_env, :, :].reshape(-1, 2, 13)
                t0_cp = next_norm_cp[:, start_env:end_env, :].reshape(-1, 6)
                
                with torch.no_grad():
                    next_val = agent.get_value(t0_self, t0_tm, t0_en, t0_cp).flatten()
                
                # GAE
                b_rewards = batch['rewards']
                b_dones = batch['dones']
                b_values = batch['values']
                
                adv = torch.zeros_like(b_rewards)
                lastgaelam = 0
                for t in reversed(range(self.current_num_steps)):
                    if t == self.current_num_steps - 1:
                        nextnonterminal = 1.0 - b_dones[t]
                        nextvalues = next_val
                    else:
                        nextnonterminal = 1.0 - b_dones[t+1]
                        nextvalues = b_values[t+1]
                    delta = b_rewards[t] + GAMMA * nextvalues * nextnonterminal - b_values[t]
                    adv[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                
                returns = adv + b_values
                
                # Flatten
                # stored batches were [NUM_STEPS, rows_per_agent, ...]
                b_obs_s = batch['self_obs'].reshape(-1, 14)
                b_obs_tm = batch['teammate_obs'].reshape(-1, 13)
                b_obs_en = batch['enemy_obs'].reshape(-1, 2, 13)
                b_obs_c = batch['cp_obs'].reshape(-1, 6)
                b_act   = batch['actions'].reshape(-1, 4)
                b_logp  = batch['logprobs'].reshape(-1)
                b_adv   = adv.reshape(-1)
                b_ret   = returns.reshape(-1)
                
                # PPO Update
                agent_samples = b_obs_s.size(0)
                inds = np.arange(agent_samples)
                
                for _ in range(UPDATE_EPOCHS):
                    np.random.shuffle(inds)
                    for st in range(0, agent_samples, MINIBATCH_SIZE):
                        ed = st + MINIBATCH_SIZE
                        mb = inds[st:ed]
                        
                        
                        # Already Normalised in Collection phase!
                        # Verify: t0_self = norm_self... -> Batch stores normalized.
                        # So we do NOT normalize again here.
                        
                        b_s_norm = b_obs_s[mb]
                        b_tm_norm = b_obs_tm[mb]
                        b_en_norm = b_obs_en[mb]
                        b_c_norm = b_obs_c[mb]
                        
                        
                        # Get Values (for RND logging if needed, but mainly for PPO)
                        
                        
                        # Role Regularization (Diversity)
                        use_div = (self.env.curriculum_stage >= STAGE_TEAM)
                        
                        if use_div:
                             _, newlog, ent, newval, divergence = agent.get_action_and_value(
                                b_s_norm, b_tm_norm, b_en_norm, b_c_norm, b_act[mb], compute_divergence=True
                             )
                        else:
                             divergence = torch.tensor(0.0, device=self.device)
                             _, newlog, ent, newval = agent.get_action_and_value(
                                b_s_norm, b_tm_norm, b_en_norm, b_c_norm, b_act[mb], compute_divergence=False
                             )
                        
                        newval = newval.flatten()
                        ratio = (newlog - b_logp[mb]).exp()
                        
                        mb_adv = b_adv[mb]
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        
                        pg1 = -mb_adv * ratio
                        current_clip = self.population[i].get('clip_range', CLIP_RANGE)
                        pg2 = -mb_adv * torch.clamp(ratio, 1-current_clip, 1+current_clip)
                        pg_loss = torch.max(pg1, pg2).mean()
                        
                        v_loss = 0.5 * ((newval - b_ret[mb])**2).mean()

                        
                        # Use Agent's Specific Entropy Coefficient
                        current_ent_coef = self.population[i].get('ent_coef', ENT_COEF)
                        div_loss = divergence.mean()
                        loss = pg_loss - current_ent_coef * ent.mean() + VF_COEF * v_loss - DIV_COEF * div_loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                        
                        # Update RND (Intrinsic Curiosity)
                        # We use the normalized observations of this batch to update the predictor
                        # CRITICAL: Detach b_s_norm to prevent backward() from trying to access PPO graph nodes already freed.
                        rnd_loss = self.rnd.update(b_s_norm.detach())
                        
                total_loss += loss.item()

            # Stats & Evolution
            global_step += NUM_ENVS * self.current_num_steps
            elapsed = time.time() - start_time
            sps = int((NUM_ENVS * self.current_num_steps) / (elapsed + 1e-6))
            
            # Record Laps
            for i in range(POP_SIZE):
                # Count laps for this agent's chunk
                # laps tensor is [4096, 4]
                start = i * ENVS_PER_AGENT
                end = start + ENVS_PER_AGENT
                # Sum laps of Pod 0 in this chunk
                laps = self.env.laps[start:end, 0].sum().item()
                # BUT this is cumulative laps.
                # We want *new* laps or total progress?
                # PBT usually checks performance over the interval.
                # Let's track 'current laps' vs 'prev laps'?
                # Or just raw laps count if we reset often.
                # Actually, env laps are cumulative since reset.
                pass
                
                # Simpler: Use the stage metrics which are cumulative, but that's global.
                # We need per-agent metrics.
                # Let's read directly from env.
                # self.population[i]['laps_score'] = laps # Cumulative -> REMOVED (Now tracking incrementally)
                pass


            # ELO Updates (Post-Iteration)
            if self.env.curriculum_stage >= STAGE_LEAGUE and hasattr(self, 'opponent_agent_loaded') and self.opponent_agent_loaded:
                 # Calculate Population Win Rate vs Opponent
                 # We tracked 'wins' in population.
                 # 'wins' were attributed if winner == 0 (Pod 0).
                 # We need total matches.
                 # Since we ran NUM_STEPS * NUM_ENVS, we can estimate matches or track them?
                 # Actually 'wins' is integer count of wins.
                 # Total matches per agent = ENVS_PER_AGENT (approx, assuming 1 match per env per iter? No.)
                 # A match ends on DONE.
                 # We need to know how many DONES happened for valid win rate calculation.
                 # Let's aggregate average score?
                 # SB3 typically uses Outcome: 1 for Win, 0 for Loss.
                 # We just aggregate all wins across population and treat as "Population vs Opponent" match?
                 
                 # Simpler: For each elite/leader, update their ELO vs Opponent.
                 # Updating for whole population might be noisy.
                 # Let's update for the LEADER.
                 
                 leader = self.population[self.leader_idx]
                 # We need specific match stats for Leader vs Opponent.
                 # We have leader['wins']. Does it normalize by matches? No.
                 # We need leader['matches_played'] or similar.
                 # We didn't track matches_played explicitly per agent in the loop above.
                 # BUT, we can infer it or just skip ELO update for now until we track it strictly?
                 
                 # Let's add 'matches' tracking in the loop quickly?
                 # Yes, let's do it below.
                 pass

            # ELO & League Stats Updates
            league_stats = None
            
            if self.env.curriculum_stage >= STAGE_DUEL and hasattr(self, 'opponent_agent_loaded') and self.opponent_agent_loaded:
                 # 1. Update Payoff Matrix for Leader
                 leader = self.population[self.leader_idx]
                 leader_id_str = f"gen_{self.generation}_agent_{leader['id']}"
                 
                 # Calculate Delta Stats (Wins in this iteration vs THIS opponent)
                 # We snapshot start stats after opponent load
                 iter_wins = leader['wins'] - self.iteration_start_wins
                 iter_matches = leader['matches'] - self.iteration_start_matches
                 
                 if iter_matches > 0:
                     win_rate = iter_wins / iter_matches
                     # Update PFSP Payoff Matrix
                     self.league.update_match_result(leader_id_str, self.current_opponent_id, win_rate)
                 else:
                     win_rate = 0.0 # No matches finished?
                 
                 # Calculate stats to show
                 # "wr" is historical (from matrix), or current? 
                 # Let's show the Cumulative Matrix WR if available, else current.
                 hist_wr = self.league.get_win_rate(leader_id_str, self.current_opponent_id)
                 
                 league_stats = {
                     "matches_played": iter_matches,
                     "wins": iter_wins,
                     "win_rate": win_rate, 
                     "opponent_id": self.current_opponent_id,
                     "registry_count": len(self.league.registry)
                 }
                 
                 self.log(f"🏆 League: {leader_id_str} vs {self.current_opponent_id} | WR: {win_rate:.2f} (Hist: {hist_wr:.2f})")

            # Evolution Check
            if self.iteration % self.current_evolve_interval == 0:
                self.evolve_population()
            
            # Logging
            self.log_iteration_summary(global_step, sps, current_tau, total_loss/POP_SIZE)
            
            # Construct simple line for telemetry log/frontend if needed (backward compat)
            leader = self.population[self.leader_idx]
            l_eff = leader.get('ema_efficiency')
            if l_eff is None: l_eff = 0.0
            log_line = f"Step: {global_step} | SPS: {sps} | Gen: {self.generation} | Leader Eff: {l_eff:.1f}"
            # self.log(log_line) # Suppressed to avoid double printing
            
            if telemetry_callback:
                # Pass league_stats as a new argument
                telemetry_callback(global_step, sps, 0, 0, self.current_win_rate, self.telemetry_env_indices[0], total_loss/POP_SIZE, log_line, False, league_stats=league_stats)

            start_time = time.time()
            
            # Save Checkpoint (Leader)
            if self.iteration % 50 == 0:
                # Save leader
                leader_agent = self.population[self.leader_idx]['agent']
                torch.save(leader_agent.state_dict(), f"data/checkpoints/model_gen{self.generation}_best.pt")

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train_loop()
