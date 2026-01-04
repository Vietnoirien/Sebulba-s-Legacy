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
import shutil

# Import Env and Constants
from simulation.env import (
    PodRacerEnv, 
    RW_WIN, RW_LOSS, RW_CHECKPOINT, RW_CHECKPOINT_SCALE, 
    RW_PROGRESS, RW_MAGNET, RW_PROXIMITY, RW_COLLISION_RUNNER, RW_COLLISION_BLOCKER, 
    RW_STEP_PENALTY, RW_ORIENTATION, RW_WRONG_WAY, RW_COLLISION_MATE,
    DEFAULT_REWARD_WEIGHTS
)
from models.deepsets import PodAgent
from training.self_play import LeagueManager
from training.normalization import RunningMeanStd
from training.evolution import calculate_novelty, fast_non_dominated_sort, calculate_crowding_distance, lexicographic_sort
from training.rnd import RNDModel
from config import *
from training.curriculum.manager import CurriculumManager

# Hyperparameters and Constants moved to config.py


class PPOTrainer:

    def __init__(self, device='cuda', logger_callback=None):
        self.config = TrainingConfig()
        self.curriculum_config = CurriculumConfig()
        self.curriculum = CurriculumManager(self.curriculum_config)
        # Sync Initial Stage Config
        # We must create env first, then sync
        self.telemetry_session = None
        
        # Override device if provided
        if device: self.config.device = device
        
        self.device = torch.device(self.config.device)
        self.env = PodRacerEnv(self.config.num_envs, device=self.device)
        self.env.set_stage(self.curriculum.current_stage_id, self.curriculum.current_stage.get_env_config())
        self.logger_callback = logger_callback
        
        # Population Initialization
        self.population = []
        self.generation = 0
        self.iteration = 0
        
        # Reward Weights [TotalEnvs, 15] - Per environment weights
        # 15 = Win, Loss, CP, Scale, Progress, Runner, Blocker, StepPen, Orient, WrongWay, Mate, Prox, Magnet, Rank, Lap
        self.reward_weights_tensor = torch.zeros((self.config.num_envs, 15), device=self.device)
        
        # Normalization
        self.rms_self = RunningMeanStd((15,), device=self.device)
        self.rms_ent = RunningMeanStd((13,), device=self.device)
        self.rms_cp = RunningMeanStd((6,), device=self.device)
        
        # RND Intrinsic Curiosity
        # Input: Normalized Self Obs (14)
        self.rnd = RNDModel(input_dim=15, device=self.device)
        self.rnd_coef = 0.01 # PPO Intrinsic Coefficient
        
        # Reward Normalization
        self.rms_ret = RunningMeanStd((1,), device=self.device)
        self.rms_ret = RunningMeanStd((1,), device=self.device)
        self.returns_buffer = torch.zeros((self.config.num_envs, 4), device=self.device)
        
        # Calculated Configs
        num_exploiters = int(self.config.pop_size * self.config.exploiter_ratio)
        split_index = self.config.pop_size - num_exploiters
        
        for i in range(self.config.pop_size):
            agent = PodAgent().to(self.device)
            optimizer = optim.Adam(agent.parameters(), lr=self.config.lr, eps=1e-5)
            
            # Initial Reward Config (Clone Default)
            # Add some noise for initial diversity?
            weights = DEFAULT_REWARD_WEIGHTS.copy()
            # Randomize slightly?
            if i > 0:
                 # Mutate orientation and velocity slightly
                 weights[RW_ORIENTATION] *= random.uniform(0.8, 1.2)
                 weights[RW_PROGRESS] *= random.uniform(0.8, 1.2)
            
            self.population.append({
                'id': i,
                'type': 'exploiter' if i >= split_index else 'main', # Dynamic Exploiter Split
                'agent': agent,
                'optimizer': optimizer,
                'weights': weights, # Python Dict for mutation logic
                'lr': self.config.lr,
                'ent_coef': self.config.ent_coef, # Individual Entropy Coefficient
                'clip_range': self.config.clip_range, # Individual Clip Range
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
                'behavior_buffer': torch.zeros(4, device=self.device),
                
                # Nursery Metrics (Inverse Distance)
                'accum_dist_fraction': 0.0,
                'accum_dist_count': 0.0,
                'nursery_score': 0.0
            })
            
            # Fill Tensor
            start_idx = i * self.config.envs_per_agent
            end_idx = start_idx + self.config.envs_per_agent
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
        self.ema_alpha = self.config.ema_alpha

        # Dynamic Hyperparameters
        self.current_num_steps = self.config.num_steps # Start with Stage 0 default
        self.current_evolve_interval = self.config.evolve_interval # Start with Stage 0 default
        self.current_active_pods_count = 1 # Start with Stage 0 default
        self.agent_batches = [] 
        self.pareto_indices = [] # Track Rank 0 agents
        
        # Difficulty Adjustment State
        self.failure_streak = 0
        self.grad_consistency_counter = 0
        self.team_spirit = 0.0 # Blending factor for rewards (0.0=Selfish, 1.0=Cooperative)
        self.current_win_rate = 0.0 # Persistent Win Rate for Telemetry
        
        # TRANSITION CONFIGURATION
        # Logic moved to CurriculumConfig, accessed via self.curriculum_config
        
        # Performance Buffer for Nursery Metrics (Avoid loop sync)
        # [PopSize, 2] -> [SumFraction, Count]
        self.nursery_metrics_buffer = torch.zeros((self.config.pop_size, 2), device=self.device)
        
        # Allocate Initial Buffers
        self.allocate_buffers()
        
        # Time Tracking
        self.train_start_time = time.time()
        self.last_log_time = time.time()
        
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
        num_active_per_agent_step = num_active_pods * self.config.envs_per_agent
        
        self.agent_batches = []
        for _ in range(self.config.pop_size):
             self.agent_batches.append({
                 'self_obs': torch.zeros((self.current_num_steps, num_active_per_agent_step, 15), device=self.device),
                 'teammate_obs': torch.zeros((self.current_num_steps, num_active_per_agent_step, 13), device=self.device),
                 'enemy_obs': torch.zeros((self.current_num_steps, num_active_per_agent_step, 2, 13), device=self.device),
                 'cp_obs': torch.zeros((self.current_num_steps, num_active_per_agent_step, 6), device=self.device),
                 'actions': torch.zeros((self.current_num_steps, num_active_per_agent_step, 4), device=self.device),
                 'rewards': torch.zeros((self.current_num_steps, num_active_per_agent_step), device=self.device),
                 'dones': torch.zeros((self.current_num_steps, num_active_per_agent_step), device=self.device),
                 'logprobs': torch.zeros((self.current_num_steps, num_active_per_agent_step), device=self.device),
                 'values': torch.zeros((self.current_num_steps, num_active_per_agent_step), device=self.device),
             })

        # Resize Nursery Buffer if pop size changed (unlikely but safe)
        if self.nursery_metrics_buffer.shape[0] != self.config.pop_size:
             self.nursery_metrics_buffer = torch.zeros((self.config.pop_size, 2), device=self.device)
    
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
                 "progress": w[RW_PROGRESS], "collision_runner": w[RW_COLLISION_RUNNER],
                 "collision_blocker": w[RW_COLLISION_BLOCKER], "step_penalty": w[RW_STEP_PENALTY],
                 "orientation": w[RW_ORIENTATION], "wrong_way_alpha": w[RW_WRONG_WAY],
                 "collision_mate": w[RW_COLLISION_MATE], "proximity": w[RW_PROXIMITY],
                 "magnet": w[RW_MAGNET]
             }
        }

    async def send_telemetry(self, step, fps_phys, fps_train, reward, win_rate, env_idx=0):
        # Identify Agent
        agent_idx = env_idx // self.config.envs_per_agent
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
            
            if self.telemetry_session is None or self.telemetry_session.closed:
                self.telemetry_session = aiohttp.ClientSession()
            
            await self.telemetry_session.post('http://localhost:8000/api/telemetry', json=payload)
        except Exception:
            pass

    async def close(self):
        if self.telemetry_session and not self.telemetry_session.closed:
            await self.telemetry_session.close()
            self.telemetry_session = None

    def check_curriculum(self):
        self.curriculum.update(self)
        self.update_step_penalty_annealing()

    def get_active_pods(self):
        return self.curriculum.get_active_pods()

    def evolve_population(self):
        self.log(f"--- Evolution Step (Gen {self.generation}) ---")
        
        # 1. Update Metrics & EMA Calculation
        # Proficiency = AvgSteps + Penalty/(Sqrt(Hits)+1). Lower is Better.
        PENALTY_CONST = self.config.proficiency_penalty_const
        
        # Collect raw values for debug logging
        debug_raw_fitness = []
        
        # Retrieve Nursery Metrics from Tensor
        nursery_data = self.nursery_metrics_buffer.cpu().numpy() # [Pop, 2]
        
        for i, p in enumerate(self.population):
            # Sync Nursery Data
            # Note: We accumulate into p['accum_dist_fraction'] for persistence?
            # Or just overwrite? 
            # It's reset every generation.
            p['accum_dist_fraction'] = nursery_data[i, 0]
            p['accum_dist_count'] = nursery_data[i, 1]
        
        for p in self.population:
            # Nursery Score Calculation
            if p['accum_dist_count'] > 0:
                avg_dist_frac = p['accum_dist_fraction'] / p['accum_dist_count']
            else:
                avg_dist_frac = 0.0 # Worst case (max distance)
            
            # Formula: (Hits * 50000) + (AvgProgress * 20000)
            # Hits must dominate. Max Progress = 1.0 * 20000.
            # So 1 Hit (50000) > Max Progress (20000).
            p['nursery_score'] = (p['total_cp_hits'] * 50000.0) + (avg_dist_frac * 20000.0)

            # Efficiency (Avg Steps per Checkpoint)

            if p['total_cp_hits'] > 0:
                raw_avg = p['total_cp_steps'] / p['total_cp_hits']
            else:
                raw_avg = 999.0 # Penalty
            
            # Proficiency Score (Raw)
            prof_score = raw_avg + (PENALTY_CONST / (np.sqrt(p['total_cp_hits'] + 1)))
            if i < 5: # Debug first 5 agents
                 self.log(f"DEBUG EFFICIENCY: Agent {p['id']} | Steps {p['total_cp_steps']} | Hits {p['total_cp_hits']} | RawAvg {raw_avg:.1f} | Score {prof_score:.1f}")
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
            
        # 3. Hybrid Selection Strategy
        if self.env.curriculum_stage <= STAGE_SOLO:
             # Strict Lexicographic Sort (Nursery & Solo)
             # Returns [[best], [2nd], ...]
             fronts = lexicographic_sort(self.population, self.env.curriculum_stage)
             objectives_np = None 
        else:
             # NSGA-II for Duel+ (Strategy Diversity)
             objectives_list = []
             for p in self.population:
                  # Delegate to Stage definition
                  # This allows each stage to define its own Multi-Objective strategy
                  objs = self.curriculum.current_stage.get_objectives(p)
                  objectives_list.append(objs)
                  
             objectives_np = np.array(objectives_list)
             fronts = fast_non_dominated_sort(objectives_np)
        
        # Calculate Crowding Distance
        if objectives_np is not None:
             crowding = calculate_crowding_distance(objectives_np, fronts)
        else:
             # For Lexicographic, rank is strict, so crowding is irrelevant for selection.
             # Set to 0.0
             crowding = np.zeros(self.config.pop_size)
        
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
             remaining = [self.population[i] for i in range(self.config.pop_size) if i not in front0_indices]
             # Sort remaining by Rank ASC, Crowding DESC
             remaining.sort(key=lambda x: (x['rank'], -x['crowding']))
             elites.extend(remaining[: 2 - len(elites)])
        
        # Update Pareto Indices (Rank 0) for Telemetry
        self.pareto_indices = [p['id'] for p in self.population if p.get('rank') == 0]
        # Logging
        # Logging
        elite_ids = [p['id'] for p in elites]
        
        if self.env.curriculum_stage <= STAGE_SOLO:
             # Lexicographic Logging
             self.log(f"Population Sorted: {len(fronts)} ranks (Strict)")
             self.log(f"Top Elites: {elite_ids}")
             
             if self.env.curriculum_stage == STAGE_NURSERY:
                  cons = elites[0]['ema_consistency']
                  nov = elites[0]['novelty_score']
                  self.log(f"Best Agent Stats | Cons: {cons:.1f} | Nov: {nov:.2f}")
             elif self.env.curriculum_stage == STAGE_SOLO:
                  eff = elites[0]['ema_efficiency']
                  cons = elites[0]['ema_consistency']
                  wins = elites[0].get('ema_wins', 0.0)
                  nov = elites[0]['novelty_score']
                  score = cons - eff
                  self.log(f"Best Agent Stats | Eff: {eff:.1f} | Cons: {cons:.1f} | Wins: {wins:.1%} | Nov: {nov:.2f} | Score: {score:.1f}")

        else:
             # NSGA-II Logging
             self.log(f"Pareto Fronts (Quality Gated > 20% WR): {[len(f) for f in fronts]}")
             self.log(f"Elites (Rank 0, Crowded): {elite_ids}")
             
             if self.env.curriculum_stage == STAGE_DUEL:
                  # Duel: Wins & Novelty
                  self.log(f"Elite (Crowded) Stats | Wins: {elites[0]['ema_wins']:.1%} | Nov: {elites[0]['novelty_score']:.2f}")
             elif self.env.curriculum_stage == STAGE_NURSERY:
                  # Fallback (Should not happen given if check above)
                  pass 
             else:
                  # League/Team
                  self.log(f"Elite (Crowded) Stats | Wins: {elites[0]['ema_wins']:.1%} | Nov: {elites[0]['novelty_score']:.2f}")

        # 6. Tournament Selection & Replacement
        # Identify Culls (Bottom 25% by Rank/Crowding)
        # Full sort of population
        # Key: (Rank ASC, Crowding DESC)
        
        # We want to keep Lower Rank (0 is best), Higher Crowding.
        sorted_pop_indices = list(range(self.config.pop_size))
        sorted_pop_indices.sort(key=lambda i: (self.population[i]['rank'], -self.population[i]['crowding']))
        
        # Dynamic Culling Rate: 50% for Nursery (to clear trash), 25% for others
        cull_ratio = 0.50 if self.env.curriculum_stage == STAGE_NURSERY else 0.25
        num_culls = max(2, int(self.config.pop_size * cull_ratio))
        cull_indices = sorted_pop_indices[-num_culls:]
        
        # FIX: Restrict Parent Pool to Top 25% Elites ONLY
        # Previously, anyone not culled (Top 75%) could be a parent, allowing mediocre agents to reproduce.
        num_parents = max(2, int(self.config.pop_size * 0.25))
        parent_candidates = sorted_pop_indices[:num_parents]
        
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
            loser['ent_coef'] = parent.get('ent_coef', self.config.ent_coef)
            
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
                
            # Reset Optimizer logic updated:
            # We want to PRESERVE Momentum to avoid "lobotomy".
            # Solution: Create new optimizer (linked to new parameters), then copy state.
            
            loser['optimizer'] = optim.Adam(loser['agent'].parameters(), lr=loser['lr'], eps=1e-5)
            
            # Copy Optimizer State (Momentum) from Parent
            parent_opt = parent['optimizer']
            loser_opt = loser['optimizer']
            
            # Adam state is mapped by Parameter object.
            # We need to map Parent Param -> State -> Loser Param -> State.
            # Since architectures are identical, we can use index mapping.
            
            parent_params = list(parent['agent'].parameters())
            loser_params = list(loser['agent'].parameters())
            
            for p_src, p_dst in zip(parent_params, loser_params):
                if p_src in parent_opt.state:
                    # Deep copy the state (exp_avg, exp_avg_sq, step) to fresh buffer
                    # We must ensure tensors are on correct device (should be same device)
                    src_state = parent_opt.state[p_src]
                    dst_state = copy.deepcopy(src_state) # Safe copy
                    loser_opt.state[p_dst] = dst_state
            
            if random.random() < 0.3:
                loser['ent_coef'] *= random.uniform(0.8, 1.2)
                loser['ent_coef'] = max(0.0001, min(0.1, loser['ent_coef']))
                
            if random.random() < 0.3:
                loser['clip_range'] *= random.uniform(0.8, 1.2)
                loser['clip_range'] = max(0.05, min(0.4, loser['clip_range']))

            self.log(f"Agent {loser['id']} (Rank {loser['rank']}) replaced by clone of {parent['id']} (Rank {parent['rank']})")
             # Update Global Tensor
            start_idx = loser['id'] * self.config.envs_per_agent
            end_idx = start_idx + self.config.envs_per_agent
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
            p['accum_dist_fraction'] = 0.0
            p['accum_dist_count'] = 0.0
        
        # Reset Nursery Buffer
        self.nursery_metrics_buffer.zero_()
        
        # Reset Nursery Buffer
        self.nursery_metrics_buffer.zero_()

            
        self.generation += 1
        # --- LEADER SELECTION (Performance Based) ---
        # Select strictly best performer from Front 0 (or population if empty)
        candidates = fronts[0] 
        if not candidates: candidates = range(len(self.population))
        
        # --- LEADER SELECTION (Performance Based) ---
        # Select strictly best performer from Front 0 (or population if empty)
        candidates = fronts[0] 
        if not candidates: candidates = range(len(self.population))
        
        if self.env.curriculum_stage == STAGE_NURSERY:
             # Stage 0: Priority Consistency, then Nursery Score, then Novelty
             # This ensures we always pick the one with most hits, 
             # and break ties with distance, then with diversity.
             best_guy = max(candidates, key=lambda i: (
                 self.population[i].get('ema_consistency', 0.0) or 0.0,
                 self.population[i].get('nursery_score', 0.0),
                 self.population[i].get('novelty_score', 0.0)
             ))
             
        elif self.env.curriculum_stage == STAGE_SOLO:
             # Combined Metric: Wins, then Consistency, then Efficiency (Lower is better)
             def combined_score(idx):
                 p = self.population[idx]
                 wins = p.get('ema_wins', 0.0) if p.get('ema_wins') is not None else 0.0
                 cons = p.get('ema_consistency', 0.0) if p.get('ema_consistency') is not None else 0.0
                 eff = p.get('ema_efficiency', 999.0) if p.get('ema_efficiency') is not None else 999.0
                 return (wins, cons, -eff)
                 
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

        # --- Auto-Flush Old Generations ---
        try:
            MAX_KEEP = 5
            gen_root = "data/generations"
            existing_gens = []
            
            # Scan
            if os.path.exists(gen_root):
                for d in os.listdir(gen_root):
                    path = os.path.join(gen_root, d)
                    if d.startswith("gen_") and os.path.isdir(path):
                        try:
                            g_num = int(d.split('_')[1])
                            existing_gens.append((g_num, path))
                        except:
                            pass
            
            # Sort and Delete
            existing_gens.sort(key=lambda x: x[0]) # Ascending
            
            if len(existing_gens) > MAX_KEEP:
                to_delete = existing_gens[:-MAX_KEEP]
                for g_num, path in to_delete:
                    shutil.rmtree(path)
                    self.log(f"Auto-Flush: Deleted old generation {g_num} to save space.")
                    
        except Exception as e:
            self.log(f"Auto-Flush Warning: {e}")
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
        
        # Nursery Specific Logging
        if self.env.curriculum_stage == STAGE_NURSERY:
             nurs = [p.get('nursery_score', 0.0) for p in self.population]
             avg_nur = np.mean(nurs) if nurs else 0.0
             l_nur = leader.get('nursery_score', 0.0)
             b_nur = 0.0 # Will calc below
        
        # Leader Stats

        l_eff = leader.get('ema_efficiency')
        if l_eff is None: l_eff = 999.0
        
        l_con = leader.get('ema_consistency')
        if l_con is None: l_con = 0.0
        
        l_win = leader.get('ema_wins')
        if l_win is None: l_win = 0.0
        
        l_nov = leader.get('novelty_score')
        if l_nov is None: l_nov = 0.0
        
        # Best Agent Stats (Not Column Max)
        # Find the agent that would be selected as leader (Best per Stage Criteria)
        # Re-use logic or just pick the one with best metric?
        # Let's find "Best Available" based on Stage criteria just for display.
        stage = self.env.curriculum_stage
        
        if stage == STAGE_SOLO:
            # Best is Wins -> Consistency -> Efficiency
            def best_sorter(p):
                 wins = p.get('ema_wins', 0.0) if p.get('ema_wins') is not None else 0.0
                 cons = p.get('ema_consistency', 0.0) if p.get('ema_consistency') is not None else 0.0
                 eff = p.get('ema_efficiency', 999.0) if p.get('ema_efficiency') is not None else 999.0
                 return (wins, cons, -eff)
            best_agent = max(self.population, key=best_sorter)
        else:
            # Best is Max Win Rate
            best_agent = max(self.population, key=lambda p: p.get('ema_wins', 0.0) if p.get('ema_wins') is not None else 0.0)
            
        b_eff = best_agent.get('ema_efficiency', 999.0)
        if b_eff is None: b_eff = 999.0
        b_con = best_agent.get('ema_consistency', 0.0)
        if b_con is None: b_con = 0.0
        b_win = best_agent.get('ema_wins', 0.0)
        if b_win is None: b_win = 0.0
        b_nov = best_agent.get('novelty_score', 0.0)
        
        # Format Table
        border = "=" * 80
        
        # Get Current Step Penalty (Mean, as it might vary slightly or be uniform)
        curr_step_pen = self.reward_weights_tensor[:, RW_STEP_PENALTY].mean().item()
        
        # Calculate Times
        current_time = time.time()
        iter_duration = current_time - self.last_log_time
        total_duration = current_time - self.train_start_time
        self.last_log_time = current_time
        
        # Format Iteration Duration (MM:SS)
        iter_m, iter_s = divmod(int(iter_duration), 60)
        iter_str = f"{iter_m:02d}:{iter_s:02d}"
        
        # Format Total Duration (Dd Hh Mm)
        total_m, total_s = divmod(int(total_duration), 60)
        total_h, total_m = divmod(total_m, 60)
        total_d, total_h = divmod(total_h, 24)
        
        if total_d > 0:
            total_str = f"{total_d}d {total_h:02d}h {total_m:02d}m"
        elif total_h > 0:
             total_str = f"{total_h}h {total_m:02d}m"
        else:
             total_str = f"{total_m}m {total_s}s"
        
        self.log(border)
        self.log(f" ITERATION {self.iteration} | Gen {self.generation} | Step {global_step} | SPS {sps} | Iter: {iter_str} | Total: {total_str}")
        self.log(f" Stage: {self.env.curriculum_stage} | Difficulty: {self.env.bot_difficulty:.2f} | Tau: {current_tau:.2f} | Step Pen: {curr_step_pen:.1f}")
        self.log("-" * 80)
        self.log(f" {'Metric':<15} | {'Leader':<10} | {'Pop Avg':<10} | {'Best Agt':<10}")
        self.log("-" * 80)
        self.log(f" {'Efficiency':<15} | {l_eff:<10.1f} | {avg_eff:<10.1f} | {b_eff:<10.1f}")
        self.log(f" {'Consistency':<15} | {l_con:<10.1f} | {avg_con:<10.1f} | {b_con:<10.1f}")
        self.log(f" {'Wins (EMA)':<15} | {l_win:<10.1%} | {avg_win:<10.1%} | {b_win:<10.1%}")
        self.log(f" {'Novelty':<15} | {l_nov:<10.2f} | {avg_nov:<10.2f} | {b_nov:<10.2f}")
        
        if self.env.curriculum_stage == STAGE_NURSERY:
             # Best Nursery Score
             best_nurs_agent = max(self.population, key=lambda p: p.get('nursery_score', 0.0))
             b_nur = best_nurs_agent.get('nursery_score', 0.0)
             self.log(f" {'Nursery Sc':<15} | {l_nur:<10.0f} | {avg_nur:<10.0f} | {b_nur:<10.0f} | (Novelty: {l_nov:.2f})")
             
        self.log("-" * 80)

        self.log(f" Loss: {avg_loss:.4f}")
        self.log(border)

    def update_step_penalty_annealing(self):
        """
        Anneals Step Penalty based on Curriculum Stage and Bot Difficulty.
        Stage 0 (Solo): Full Penalty (5.0)
        Stage > 0: Linear Decay based on Bot Difficulty.
        """
        base_penalty = DEFAULT_REWARD_WEIGHTS[RW_STEP_PENALTY] # 5.0
        new_val = self.curriculum.update_step_penalty(base_penalty)
         
        # Update Tensor
        self.reward_weights_tensor[:, RW_STEP_PENALTY] = new_val
        
    def train_loop(self, stop_event=None, telemetry_callback=None):
        self.log(f"Starting Evolutionary PPO (Pop: {self.config.pop_size}, Envs/Agent: {self.config.envs_per_agent})...")
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

        while global_step < self.config.total_timesteps:
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
                 
                 # GENERATE NEW MATCH ID (Forces Frontend Reset)
                 self.match_id = str(uuid.uuid4())
                 
                 obs_data = self.env.get_obs()


            # --- Dynamic Config Check ---
            current_stage = self.env.curriculum_stage
            
            # SOTA Tuning (See stage_0_tuning_report.md)
            # Nursery: Fast Evolution (1) to find movers. Steps 256.
            # Solo+: Stable Evolution (2). Steps 256.
            
            # SOTA Tuning (See stage_0_tuning_report.md)
            # SOTA Tuning (See stage_0_tuning_report.md)
            # Delegated to Stage Class (Evolve Interval)
            # target_steps removed - fixed to config.num_steps (512)
            target_evolve = self.curriculum.current_stage.target_evolve_interval
            
            # Dynamic Evolve Check (if callable/property logic is complex, might need method)
            # For Team Stage, it was dynamic: int(8 - 4 * diff).
            # We will handle this in the TeamStage property implementation.
            
            # Apply Evolve Interval
            self.current_evolve_interval = target_evolve         
            

            current_active_pods_ids = self.get_active_pods()
            active_count = len(current_active_pods_ids)
            
            needs_realloc = False
            # Apply Num Steps (Fixed)
            self.current_num_steps = self.config.num_steps
            
            if active_count != self.current_active_pods_count:
                 self.log(f"Stage Change Triggered Config Update: Active Pods {self.current_active_pods_count} -> {active_count}")
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
            num_active_per_agent = len(active_pods) * self.config.envs_per_agent
            
            # --- Collection Phase (Parallel) ---
            # Batches for each agent
            # Using self.agent_batches (allocated dynamically)
                 
            # Unpack Obs
            # obs_data is tuple (self, tm, en, cp) tensors [4, self.config.num_envs, ...]
            all_self, all_tm, all_en, all_cp = obs_data
            
            for step in range(self.current_num_steps):
                if stop_event and stop_event.is_set(): break
                # --- Global Normalization ---
                # Normalize ALL active observations at once for efficiency
                # Source: [4, self.config.num_envs, ...]
                
                # Active Pods Slicing
                # active_pods is list [0] or [0, 1] etc.
                # all_self[active_pods] returns [N_Active, self.config.num_envs, 14]
                # It copies and stacks them. Since all_self[i] is [self.config.num_envs, 14] contiguous, this is fast.
                
                raw_self = all_self[active_pods].view(-1, 15) # [N_Active * self.config.num_envs, 15]
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
                
                # The normalized tensors are flat [N_Active * self.config.num_envs, ...]
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
                # Keep normalized as [N_Active, self.config.num_envs, D]
                
                norm_self = norm_self.view(len(active_pods), self.config.num_envs, 15)
                norm_tm = norm_tm.view(len(active_pods), self.config.num_envs, 13)
                norm_en = norm_en.view(len(active_pods), self.config.num_envs, 2, 13)
                norm_cp = norm_cp.view(len(active_pods), self.config.num_envs, 6)
                
                # Now Agent i needs envs i*128 : (i+1)*128
                # across ALL active pods.
                # So we slice dim 1.
                
                for i in range(self.config.pop_size):
                    start_env = i * self.config.envs_per_agent
                    end_env = start_env + self.config.envs_per_agent
                    
                    # Slice Normalized Tensors [N_Active, Batch, D]
                    # We want [N_Active * Batch, D] for input to network
                    # But careful with ordering. DeepSets treats input as Batch independent.
                    # Flattening [N_A, B, D] -> [N_A*B, D] stacks pods: Pod0(Batch), Pod1(Batch).
                    # This is fine.
                    
                    t0_self = norm_self[:, start_env:end_env, :].reshape(-1, 15)
                    t0_tm = norm_tm[:, start_env:end_env, :].reshape(-1, 13)
                    t0_en = norm_en[:, start_env:end_env, :, :].reshape(-1, 2, 13)
                    t0_cp = norm_cp[:, start_env:end_env, :].reshape(-1, 6)
                    
                    with torch.no_grad():
                        agent = self.population[i]['agent']
                        action0, logprob0, _, value0 = agent.get_action_and_value(t0_self, t0_tm, t0_en, t0_cp)
                        
                    # Store
                    # Agent batch storage expectation:
                    # 'self_obs': [self.config.num_steps, n_active_per_agent, 14]
                    # n_active_per_agent = N_Active * self.config.envs_per_agent
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
                     # Main: Agents 0 to SPLIT_INDEX -> Envs 0 to Split * EnvsPerAgent
                     # Exploiter: Agents Split to End -> Envs ...
                     
                     # DYNAMIC EXPLOITER LOGIC:
                     # If Stage < DUEL, Disable Exploiters (All agents act as Main against Opponent/Self)
                     # Actually, in Solo/Nursery, we calculate actions normally (Line 945) against Empty/Static.
                     # This block (Line 980+) is for LEAGUE mode opponents (Pods 2/3).
                     # But wait, if Stage < LEAGUE, valid?
                     # Line 980 checks: if self.env.curriculum_stage >= STAGE_LEAGUE:
                     # So this logic DOES NOT RUN in Nursery/Solo/Duel.
                     # Conclusion: Exploiters are naturally implicitly disabled in early stages because we don't use the Opponent Logic block.
                     # EXCEPT where?
                     # Ah, 'exploiter' type is just a label. They train same as everyone else in PPO.
                     # The difference is only WHO they play against in League.
                     # So no change needed here?
                     # Logic check: In Stage 0, we just run `agent.get_action_and_value` for all agents. 
                     # They all see environment. 
                     # So "Exploiter" agents just learn normally.
                     # Correct. The User asked to disable "Exploiter Logic". 
                     # If that means "Don't treat them differently", we are good as long as we don't enter this block.
                     # BUT `split_index` is used elsewhere? 
                     # No, only here.
                     # So Stage 0/1 are safe.
                     
                     # However, to be explicit as requested:
                     current_split_ratio = self.config.exploiter_ratio
                     if self.env.curriculum_stage < STAGE_DUEL:
                         current_split_ratio = 0.0
                         
                     num_exploiters_active = int(self.config.pop_size * current_split_ratio)
                     split_idx_agent = self.config.pop_size - num_exploiters_active
                     split_idx = split_idx_agent * self.config.envs_per_agent

                     
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
                     act_e = act_e.view(2, self.config.num_envs - split_idx, 4)
                     
                     # --- COMBINE ---
                     # Concatenate along Env dim (dim 1)
                     combined_opp_act = torch.cat([act_m, act_e], dim=1) # [2, 4096, 4]
                     
                     # Map to Act Map
                     act_map[2] = combined_opp_act[0] # Pod 2
                     act_map[3] = combined_opp_act[1] # Pod 3
                
                # Construct Global Action Tensor
                env_actions = torch.zeros((self.config.num_envs, 4, 4), device=self.device)
                
                for pid, act in act_map.items():
                    env_actions[:, pid] = act

                
                # Calculate Tau (Dense Reward Annealing)
                # 0.0 (Full Dense) -> 1.0 (Full Sparse/Shaped)
                # Stage 0: 0.0
                # Stage 1: 0.5
                # Stage 2: 0.9
                current_tau = 0.0
                if self.env.curriculum_stage == STAGE_SOLO:
                    current_tau = 0.25
                elif self.env.curriculum_stage == STAGE_DUEL:
                    current_tau = 0.5
                elif self.env.curriculum_stage == STAGE_TEAM:
                    current_tau = 0.75
                elif self.env.curriculum_stage == STAGE_LEAGUE:
                    current_tau = 0.9
                
                # STEP
                # Pass Global Reward Tensor
                # info is dictionary now
                # rewards is [self.config.num_envs, 2] (Team 0, Team 1)
                # dones is [self.config.num_envs]
                rewards_all, dones, infos = self.env.step(env_actions, reward_weights=self.reward_weights_tensor, tau=current_tau, team_spirit=self.team_spirit)
                
                # --- Reward Processing ---
                # Blending is now handled in Env (Hybrid: Individual Dense + Shared Sparse)
                blended_rewards = rewards_all
                
                # --- Reward Normalization ---
                # 1. Update Returns Buffer (All 4 channels)
                dones_exp = dones.unsqueeze(1).expand(-1, 4)
                self.returns_buffer = blended_rewards + self.config.gamma * self.returns_buffer * (~dones_exp).float()
                
                # 2. Update RMS (Flattened)
                self.rms_ret.update(self.returns_buffer.view(-1))
                
                # 3. Normalize
                rew_var = torch.clamp(self.rms_ret.var, min=1e-4)
                rew_std = torch.sqrt(rew_var)
                norm_rewards = torch.clamp(blended_rewards / rew_std, -10.0, 10.0) # [B, 4]
                
                # --- Intrinsic Curiosity Update ---
                # RND Logic
                rnd_input = norm_self.reshape(-1, 15).detach() # [N_Active * self.config.num_envs, 15]
                intrinsic_rewards = self.rnd.compute_intrinsic_reward(rnd_input) # [N_Active * self.config.num_envs]
                
                # Map Intrinsic to Pods
                r_int = intrinsic_rewards.view(len(active_pods), self.config.num_envs)
                
                if self.env.curriculum_stage == STAGE_NURSERY or self.env.curriculum_stage == STAGE_SOLO:
                     # Only add to Pod 0 (active_pods[0])
                     norm_rewards[:, 0] += r_int[0] * self.rnd_coef

                # Split rewards back to agents
                # --- Vectorized Nursery Metric Tracking ---
                # Done OUTSIDE the loop to avoid 128x overhead
                agent_dists = infos.get('dist_to_next', None)
                if agent_dists is not None:
                     # agent_dists is [NumEnvs, 4] -> [Pop*EnvsPerAg, 4]
                     # We need to process all at once.
                     
                     # Extract active pod distances
                     # For Nursery/Solo, active_pods=[0].
                     # dists_active: [NumEnvs, N_active]
                     dists_active = agent_dists[:, active_pods]
                     
                     # Normalize: [NumEnvs, N_active]
                     # INCREASED CONSTANT from 5000 to 20000 to cover full map diagonal (~18300)
                     start_norm_dists = torch.clamp(1.0 - (dists_active / 20000.0), 0.0, 1.0)
                     
                     # Sum active pods per env -> [NumEnvs]
                     # If multiple pods, we sum them? Yes.
                     env_scores = start_norm_dists.sum(dim=1) 
                     env_counts = torch.tensor(len(active_pods), device=self.device).repeat(self.config.num_envs)
                     
                     # Now reduce by Agent.
                     # Reshape [Pop, EnvsPerAgent]
                     # This assumes num_envs is exactly Pop * EnvsPerAgent (always true here)
                     env_scores_reshaped = env_scores.view(self.config.pop_size, self.config.envs_per_agent)
                     env_counts_reshaped = env_counts.view(self.config.pop_size, self.config.envs_per_agent)
                     
                     # Sum across EnvsPerAgent -> [Pop]
                     agent_scores = env_scores_reshaped.sum(dim=1)
                     agent_counts = env_counts_reshaped.sum(dim=1)
                     
                     # Update Buffer
                     self.nursery_metrics_buffer[:, 0] += agent_scores
                     self.nursery_metrics_buffer[:, 1] += agent_counts

                # --- Per-Agent Stats Loop ---
                for i in range(self.config.pop_size):
                    start_env = i * self.config.envs_per_agent
                    end_env = start_env + self.config.envs_per_agent
                    
                    r_chunks = []
                    d_chunks = []
                    
                    raw_sum = 0
                    raw_count = 0
                    
                    d_slice = dones[start_env:end_env]
                    
                    for p_idx, pid in enumerate(active_pods):
                        r_slice = norm_rewards[start_env:end_env, pid]
                        r_chunks.append(r_slice)
                        d_chunks.append(d_slice)
                        
                        # Logging Raw
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
                    
                    # Checkpoints
                    # Filter infos first
                    agent_infos_laps = infos['laps_completed'][start_env:end_env] # [128, 4]
                    agent_infos_cps = infos['checkpoints_passed'][start_env:end_env] # [128, 4]
                    agent_start_streak = infos['current_streak'][start_env:end_env] # [128, 4]
                    
                    # Mask by active pods
                    active_laps = 0
                    active_cps = 0
                    for pid in active_pods:
                         active_laps += agent_infos_laps[:, pid].sum().item()
                         passed_mask = (agent_infos_cps[:, pid] > 0)
                         streak_mask = (agent_start_streak[:, pid] > 0)
                         active_cps += (passed_mask & streak_mask).sum().item()
                         
                    self.population[i]['laps_score'] += active_laps
                    self.population[i]['checkpoints_score'] += active_cps
                    
                    # Accumulate Role Metrics
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
                    start_streak = agent_start_streak 
                    start_steps = infos['cp_steps'][start_env:end_env] 
                    
                    # Max Streak
                    current_max = start_streak.max().item()
                    if current_max > self.population[i]['max_streak']:
                        self.population[i]['max_streak'] = current_max
                        
                    # Efficiency
                    # FILTER CP1 FARMERS: Only record efficiency for streak > 0
                    mask = (start_steps > 0) & (start_streak > 0)
                    if mask.any():
                        self.population[i]['total_cp_steps'] += start_steps[mask].sum().item()
                        self.population[i]['total_cp_hits'] += mask.sum().item()
                    
                    # Wins (Track from env.winners on Reset)
                    agent_dones = d_slice.bool()
                    if agent_dones.any():
                        # Global indices
                        idx_rel = torch.nonzero(agent_dones).flatten()
                        idx_global = start_env + idx_rel
                        
                        current_winners = self.env.winners[idx_global]
                        
                        win_count = (current_winners == 0).sum().item()
                        self.population[i]['wins'] += win_count
                        
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
                                 if is_done_env:
                                     done_indices = torch.nonzero(dones).flatten()
                                     if len(done_indices) > 0:
                                         candidates = done_indices.tolist()
                                         
                                         # Smart Selection
                                         # If Slot < 16 (First half), prioritize Pareto (Rank 0)
                                         # Else, random (representing Evolved Pop)
                                         selected_idx = None
                                         
                                         if t_idx < 16 and len(self.pareto_indices) > 0:
                                             # Try to find a Pareto candidate
                                             pareto_candidates = []
                                             for cand in candidates:
                                                 agent_id = cand // self.config.envs_per_agent
                                                 if agent_id in self.pareto_indices:
                                                     pareto_candidates.append(cand)
                                             
                                             if pareto_candidates:
                                                 selected_idx = random.choice(pareto_candidates)
                                         
                                         if selected_idx is None:
                                              selected_idx = random.choice(candidates)
                                         
                                         self.telemetry_env_indices[t_idx] = selected_idx
                
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
                
                for i in range(self.config.pop_size):
                    start_env = i * self.config.envs_per_agent
                    end_env = start_env + self.config.envs_per_agent
                    
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
                    
                    count = self.config.envs_per_agent * len(active_pods)
                    
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
            # Source: [4, self.config.num_envs, ...]
            
            # Global Normalize Next Obs (Fixed=True)
            # Source: [4, self.config.num_envs, ...]
            
            next_raw_self = all_self[active_pods].view(-1, 15)
            next_raw_tm = all_tm[active_pods].view(-1, 13)
            next_raw_en = all_en[active_pods].view(-1, 2, 13)
            next_raw_cp = all_cp[active_pods].view(-1, 6)
            
            next_norm_self = self.rms_self(next_raw_self, fixed=True)
            
            next_norm_tm = self.rms_ent(next_raw_tm, fixed=True)
            B_e, N_e, D_e = next_raw_en.shape
            next_norm_en = self.rms_ent(next_raw_en.view(-1, D_e), fixed=True).view(B_e, N_e, D_e)
            
            next_norm_cp = self.rms_cp(next_raw_cp, fixed=True)
            
            # View as [N_Act, self.config.num_envs, ...]
            next_norm_self = next_norm_self.view(len(active_pods), self.config.num_envs, 15)
            next_norm_tm = next_norm_tm.view(len(active_pods), self.config.num_envs, 13)
            next_norm_en = next_norm_en.view(len(active_pods), self.config.num_envs, 2, 13)
            next_norm_cp = next_norm_cp.view(len(active_pods), self.config.num_envs, 6)
            
            # --- VECTORIZED GAE ---
            # 1. Collect Global Data
            all_next_vals = []
            all_rewards = []
            all_dones = []
            all_values = []
            
            # Batch size per agent (Envs * ActivePods)
            bs_per_agent = self.config.envs_per_agent * len(active_pods)

            # Pre-compute Next Values for all agents
            for i in range(self.config.pop_size):
                agent = self.population[i]['agent']
                
                # Slice Next Val Inputs matches current loop logic
                start_env = i * self.config.envs_per_agent
                end_env = start_env + self.config.envs_per_agent
                
                t0_self = next_norm_self[:, start_env:end_env, :].reshape(-1, 15)
                t0_tm = next_norm_tm[:, start_env:end_env, :].reshape(-1, 13)
                t0_en = next_norm_en[:, start_env:end_env, :, :].reshape(-1, 2, 13)
                t0_cp = next_norm_cp[:, start_env:end_env, :].reshape(-1, 6)
                
                with torch.no_grad():
                     n_v = agent.get_value(t0_self, t0_tm, t0_en, t0_cp).flatten()
                     all_next_vals.append(n_v)
                
                # Collect buffer Refs
                batch = self.agent_batches[i]
                all_rewards.append(batch['rewards'])
                all_dones.append(batch['dones'])
                all_values.append(batch['values'])

            # 2. Global Concatenation [Time, TotalBatch]
            g_next_val = torch.cat(all_next_vals, dim=0) # [TotalBatch]
            g_rewards = torch.cat(all_rewards, dim=1)    # [Time, TotalBatch]
            g_dones = torch.cat(all_dones, dim=1)        # [Time, TotalBatch]
            g_values = torch.cat(all_values, dim=1)      # [Time, TotalBatch]
            
            # 3. Single Global GAE Loop (512 iters vs 65k)
            g_adv = torch.zeros_like(g_rewards)
            lastgaelam = 0
            
            for t in reversed(range(self.current_num_steps)):
                if t == self.current_num_steps - 1:
                    nextnonterminal = 1.0 - g_dones[t]
                    nextvalues = g_next_val
                else:
                    nextnonterminal = 1.0 - g_dones[t+1]
                    nextvalues = g_values[t+1]
                
                delta = g_rewards[t] + self.config.gamma * nextvalues * nextnonterminal - g_values[t]
                g_adv[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            
            g_returns = g_adv + g_values

            # 4. PPO Update Loop (Slicing back)
            for i in range(self.config.pop_size):
                batch = self.agent_batches[i]
                agent = self.population[i]['agent']
                optimizer = self.population[i]['optimizer']

                # Slice Global Results
                st_idx = i * bs_per_agent
                ed_idx = st_idx + bs_per_agent
                
                adv = g_adv[:, st_idx:ed_idx]
                returns = g_returns[:, st_idx:ed_idx]
                
                # Flatten
                # stored batches were [self.config.num_steps, rows_per_agent, ...]
                b_obs_s = batch['self_obs'].reshape(-1, 15)
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
                
                for _ in range(self.config.update_epochs):
                    np.random.shuffle(inds)
                    minibatch_size = self.config.batch_size // self.config.num_minibatches
                    for st in range(0, agent_samples, minibatch_size):
                        ed = st + minibatch_size
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
                        current_clip = self.population[i].get('clip_range', self.config.clip_range)
                        pg2 = -mb_adv * torch.clamp(ratio, 1-current_clip, 1+current_clip)
                        pg_loss = torch.max(pg1, pg2).mean()
                        
                        v_loss = 0.5 * ((newval - b_ret[mb])**2).mean()

                        
                        # Use Agent's Specific Entropy Coefficient
                        current_ent_coef = self.population[i].get('ent_coef', self.config.ent_coef)
                        div_loss = divergence.mean()
                        loss = pg_loss - current_ent_coef * ent.mean() + self.config.vf_coef * v_loss - self.config.div_coef * div_loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), self.config.max_grad_norm)
                        optimizer.step()
                        
                        # Update RND (Intrinsic Curiosity)
                        # We use the normalized observations of this batch to update the predictor
                        # CRITICAL: Detach b_s_norm to prevent backward() from trying to access PPO graph nodes already freed.
                        rnd_loss = self.rnd.update(b_s_norm.detach())
                        
                total_loss += loss.item()

            # Stats & Evolution
            global_step += self.config.num_envs * self.current_num_steps
            elapsed = time.time() - start_time
            sps = int((self.config.num_envs * self.current_num_steps) / (elapsed + 1e-6))
            
            # Record Laps
            for i in range(self.config.pop_size):
                # Count laps for this agent's chunk
                # laps tensor is [4096, 4]
                start = i * self.config.envs_per_agent
                end = start + self.config.envs_per_agent
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
                 # Since we ran self.config.num_steps * self.config.num_envs, we can estimate matches or track them?
                 # Actually 'wins' is integer count of wins.
                 # Total matches per agent = self.config.envs_per_agent (approx, assuming 1 match per env per iter? No.)
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
            self.log_iteration_summary(global_step, sps, current_tau, total_loss/self.config.pop_size)
            
            # Construct simple line for telemetry log/frontend if needed (backward compat)
            leader = self.population[self.leader_idx]
            l_eff = leader.get('ema_efficiency')
            if l_eff is None: l_eff = 0.0
            log_line = f"Step: {global_step} | SPS: {sps} | Gen: {self.generation} | Leader Eff: {l_eff:.1f}"
            # self.log(log_line) # Suppressed to avoid double printing
            
            if telemetry_callback:
                # Pass league_stats as a new argument
                telemetry_callback(global_step, sps, 0, 0, self.current_win_rate, self.telemetry_env_indices[0], total_loss/self.config.pop_size, log_line, False, league_stats=league_stats)

            start_time = time.time()
            
            # Save Checkpoint (Leader)
            if self.iteration % 50 == 0:
                # Save leader
                leader_agent = self.population[self.leader_idx]['agent']
                torch.save(leader_agent.state_dict(), f"data/checkpoints/model_gen{self.generation}_best.pt")

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train_loop()
