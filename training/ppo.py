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
from training.optimizers import VectorizedAdam
from torch.func import functional_call, vmap, grad

# Hyperparameters and Constants moved to config.py


class PPOTrainer:

    def __init__(self, config: Optional[TrainingConfig] = None, device='cuda', logger_callback=None):
        self.config = config if config is not None else TrainingConfig()
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
        self.reward_weights_tensor = torch.zeros((self.config.num_envs, 16), device=self.device)
        
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
        
        # Performance Optimizations
        self.use_amp = torch.cuda.is_available() and getattr(torch.cuda, 'amp', None) is not None
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda') if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.compile_model = True if self.device.type == 'cuda' else False 
        
        # Template Agent for Functional Calls (Reference architecture)
        self.template_agent = PodAgent().to(self.device)
        # self.template_agent.forward is now handled by class definition with 'method' kwarg

        for i in range(self.config.pop_size):
            agent = PodAgent().to(self.device)
            # No torch.compile here. No optimizer here.
            
            # Initial Reward Config (Clone Default)
            weights = DEFAULT_REWARD_WEIGHTS.copy()
            if i > 0:
                 weights[RW_ORIENTATION] *= random.uniform(0.8, 1.2)
                 weights[RW_PROGRESS] *= random.uniform(0.8, 1.2)
            
            self.population.append({
                'id': i,
                'type': 'exploiter' if i >= split_index else 'main',
                'agent': agent,
                # 'optimizer': optimizer, # REMOVED
                'weights': weights, 
                'lr': self.config.lr,
                'ent_coef': self.config.ent_coef,
                'clip_range': self.config.clip_range,
                'laps_score': 0,
                'checkpoints_score': 0,
                'reward_score': 0.0,
                'efficiency_score': 999.0,
                'wins': 0,
                'matches': 0,
                'max_streak': 0,
                'total_cp_steps': 0,
                'total_cp_hits': 0,
                'avg_steps': 0.0,
                'avg_runner_vel': 0.0,
                'avg_blocker_dmg': 0.0,
                
                'ema_efficiency': None,
                'ema_consistency': None,
                'ema_wins': None,
                'ema_laps': None,
                'ema_runner_vel': None,
                'ema_blocker_dmg': None,
                'ema_dist': None,
                
                'behavior_buffer': torch.zeros(4, device=self.device),
                
                'accum_dist_fraction': 0.0,
                'accum_dist_count': 0.0,
                'nursery_score': 0.0
            })
            
            # Match ID (Unique per agent lifetime)

            # Fill Tensor
            start_idx = i * self.config.envs_per_agent
            end_idx = start_idx + self.config.envs_per_agent
            for k, v in weights.items():
                self.reward_weights_tensor[start_idx:end_idx, k] = v

        # Initialize Stacked Params and Vectorized Optimizer
        self.init_vectorized_population()

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
        
        # Decide Active Pods Max (Assume max 2 for buffer sizing to be safe, or separate?)
        # Buffers are per AGENT.
        # Envs per agent = 128.
        # Max pods per env = 2 (League).
        
        num_active_pods = self.current_active_pods_count
            
        # Total batch dimension = Steps * EnvsPerAgent * NumActivePods
        num_active_per_agent_step = num_active_pods * self.config.envs_per_agent
        
        # Log only if significantly changed or first run
        if not hasattr(self, 'last_buffer_shape') or self.last_buffer_shape != (self.current_num_steps, num_active_per_agent_step):
             self.log(f"Allocating buffers for {self.current_num_steps} steps per iteration. Active Pods: {self.current_active_pods_count}")
             self.last_buffer_shape = (self.current_num_steps, num_active_per_agent_step)
        
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
                 # LSTM States [Chunks, Batch, Layers, Hidden] - Sparse Storage
                 'actor_h': torch.zeros((self.current_num_steps // self.config.seq_length + 1, num_active_per_agent_step, 1, self.config.lstm_hidden_size), device=self.device),
                 'actor_c': torch.zeros((self.current_num_steps // self.config.seq_length + 1, num_active_per_agent_step, 1, self.config.lstm_hidden_size), device=self.device),
                 'critic_h': torch.zeros((self.current_num_steps // self.config.seq_length + 1, num_active_per_agent_step, 1, self.config.lstm_hidden_size), device=self.device),
                 'critic_c': torch.zeros((self.current_num_steps // self.config.seq_length + 1, num_active_per_agent_step, 1, self.config.lstm_hidden_size), device=self.device),
             })

        if self.nursery_metrics_buffer.shape[0] != self.config.pop_size:
             self.nursery_metrics_buffer = torch.zeros((self.config.pop_size, 2), device=self.device)
             
        # Behavior Stats Tensor [Pop, 4] (Speed, Steer, Dist, Count)
        if not hasattr(self, 'behavior_stats_tensor') or self.behavior_stats_tensor.shape[0] != self.config.pop_size:
             self.behavior_stats_tensor = torch.zeros((self.config.pop_size, 4), device=self.device)
        else:
             self.behavior_stats_tensor.zero_()
    
    def init_vectorized_population(self):
        """
        Consolidates individual agent parameters into a single stacked TensorDict.
        Initializes VectorizedAdam.
        """
        # 1. Stack Parameters
        first_agent = self.population[0]['agent']
        # We use strict ordering from named_parameters keys
        param_names = [n for n, _ in first_agent.named_parameters()]
        
        self.stacked_params = {}
        # Cache references for fast syncing
        self.agent_param_refs = [dict(p['agent'].named_parameters()) for p in self.population]
        
        for name in param_names:
            # Collect from all agents
            all_p = [self.agent_param_refs[i][name] for i in range(self.config.pop_size)]
            self.stacked_params[name] = torch.stack(all_p).detach().requires_grad_(True)
            
        # 2. Stack Buffers
        buffer_names = [n for n, _ in first_agent.named_buffers()]
        self.stacked_buffers = {}
        self.agent_buffer_refs = [dict(p['agent'].named_buffers()) for p in self.population]
        
        for name in buffer_names:
            all_b = [self.agent_buffer_refs[i][name] for i in range(self.config.pop_size)]
            self.stacked_buffers[name] = torch.stack(all_b)
            
        # 3. Collect LRs
        lrs = torch.tensor([p['lr'] for p in self.population], device=self.device)
        
        # 4. Initialize Vectorized Optimizer
        self.vectorized_adam = VectorizedAdam(self.stacked_params, lrs, eps=1e-5)
        
        # 5. Compile Functional Wrapper (Optional but good)
        # self.vmap_inference = torch.compile(vmap(self._functional_forward, ...))?
        
        self.log(f"Vectorized Population Initialized: {len(self.stacked_params)} param tensors stacked.")

    def sync_vectorized_to_agents(self):
        """
        Copies stacked params back to individual agent instances.
        Required for checkpointers, league manager, and evolution that rely on agent.state_dict().
        """
        with torch.no_grad():
            Pop = self.config.pop_size
            
            for name, stacked_p in self.stacked_params.items():
                for i in range(Pop):
                     # In-place copy to existing parameter tensor
                     self.agent_param_refs[i][name].copy_(stacked_p[i])
                     
            for name, stacked_b in self.stacked_buffers.items():
                for i in range(Pop):
                     self.agent_buffer_refs[i][name].copy_(stacked_b[i])

    def sync_agents_to_vectorized(self):
        """
        Copies individual agent params TO the stacked vectorized tensors.
        Must be called after loading checkpoints or manually modifying agents.
        """
        with torch.no_grad():
            Pop = self.config.pop_size
            
            # Sync Params
            for name, stacked_p in self.stacked_params.items():
                # We can iterate or stack-copy. Stack copy might be cleaner but slower?
                # stacked_p is [Pop, ...].
                # We can construct a temp stack and copy it in?
                # Or just loop. Loop 128 is fast.
                for i in range(Pop):
                    stacked_p[i].copy_(self.agent_param_refs[i][name])
            
            # Sync Buffers
            for name, stacked_b in self.stacked_buffers.items():
                for i in range(Pop):
                    stacked_b[i].copy_(self.agent_buffer_refs[i][name])
        
        self.log("Synced Agents -> Vectorized Stack.")

    def broadcast_checkpoint(self, state_dict):
        """
        Loads a state_dict (single agent) into ALL agents in the population 
        and syncs to vectorized stack.
        """
        self.log(f"Broadcasting checkpoint to all {self.config.pop_size} agents...")
        for p in self.population:
            p['agent'].load_state_dict(state_dict)
            # Reset Optimizer Momentums? Maybe.
            # But let's assume we are loading for Evaluation mostly.
            
        self.sync_agents_to_vectorized()

    def _functional_inference(self, params, buffers, s, tm, en, cp, actor_h, actor_c, critic_h, critic_c):
        """
        Functional wrapper for vmap inference.
        """
        # Reconstruct tuple state
        # Input h, c are [Batch, Layers, Hidden] (Permuted by PPO buffer format [B, L, H])
        # Wait, PPO buffer is [Pop, B, L, H]. vmap slices Pop -> [B, L, H].
        # But we need [Layers, Batch, H] for LSTM forward (standard pytorch layout).
        # So we permute (1, 0, 2).
        
        actor_state = (actor_h.permute(1,0,2).contiguous(), actor_c.permute(1,0,2).contiguous())
        critic_state = (critic_h.permute(1,0,2).contiguous(), critic_c.permute(1,0,2).contiguous())
        
        # Output: action, logprob, ent, val, states
        out = functional_call(self.template_agent, (params, buffers), (s, tm, en, cp), kwargs={'actor_state': actor_state, 'critic_state': critic_state})
        action, logprob, ent, val, states = out
        return action, logprob, ent, val, states['actor'], states['critic']

    def _functional_get_value(self, params, buffers, s, tm, en, cp, critic_h, critic_c):
        # We need critic state to get value
        critic_state = (critic_h.permute(1,0,2).contiguous(), critic_c.permute(1,0,2).contiguous())
        return functional_call(self.template_agent, (params, buffers), (s, tm, en, cp), kwargs={'method': 'get_value', 'lstm_state': critic_state})

    def _functional_loss(self, params, buffers, s, tm, en, cp, act, old_logp, old_val, adv, ret, actor_h0, actor_c0, critic_h0, critic_c0, use_div, ent_coef, clip_range, vf_coef, div_coef):
        # s, tm, en, cp are [MB, SeqLat, D]
        # h0, c0 are [MB, 1, Hidden] -> Need [Layers, MB, Hidden]
        
        # Permute states to [Layers, MB, Hidden] usually [1, MB, 64]
        # Input is [MB, 1, 64]
        a_h = actor_h0.permute(1, 0, 2).contiguous()
        a_c = actor_c0.permute(1, 0, 2).contiguous()
        c_h = critic_h0.permute(1, 0, 2).contiguous()
        c_c = critic_c0.permute(1, 0, 2).contiguous()
        
        # Forward (Sequence Mode handled by PodAgent based on input dim)
        # Forward (Sequence Mode handled by PodAgent based on input dim)
        # We disable divergence check for LSTM architecture for now as it doesn't support 'compute_divergence'
        # We disable divergence check for LSTM architecture for now as it doesn't support 'compute_divergence'
        _, logp, ent, val, states = functional_call(self.template_agent, (params, buffers), (s, tm, en, cp, act), kwargs={'actor_state': (a_h, a_c), 'critic_state': (c_h, c_c)})
        div = torch.tensor(0.0, device=s.device)
        
        next_actor_h, next_actor_c = states['actor']
        next_critic_h, next_critic_c = states['critic']
            
        # val is [MB, Seq]
        val = val.flatten() # [MB*Seq]
        # old_val is [MB, Seq] -> flatten
        old_val = old_val.flatten()
        logp = logp.flatten()
        old_logp = old_logp.flatten()
        ent = ent.flatten()
        adv = adv.flatten()
        ret = ret.flatten()
        
        ratio = (logp - old_logp).exp()
        
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
        # PPO Clip
        surr1 = -adv * ratio
        surr2 = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg_loss = torch.max(surr1, surr2).mean()
        
        # Value Loss
        v_loss = 0.5 * ((val - ret) ** 2).mean()
        
        # Entropy & Div
        # CLAMP DIVERGENCE to prevent explosion (e.g. max 10.0)
        div_clamped = torch.clamp(div.mean(), max=10.0)
        
        loss = pg_loss - ent_coef * ent.mean() + vf_coef * v_loss - div_coef * div_clamped
        
        return loss, (loss.detach(), pg_loss.detach(), v_loss.detach(), ent.mean().detach(), div_clamped.detach(), (next_actor_h, next_actor_c), (next_critic_h, next_critic_c))

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
        
        # [NEW] Team Spirit Annealing
        new_spirit = self.curriculum.update_team_spirit(self)
        if new_spirit != self.team_spirit:
             self.team_spirit = new_spirit
             # self.log(f"Config: Team Spirit updated to {self.team_spirit:.2f}") # Too spammy? No, only on change.

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
        
        # Prepare Behavior Stats (Bulk Copy to CPU)
        if hasattr(self, 'behavior_stats_tensor'):
             beh_cpu = self.behavior_stats_tensor.cpu().numpy()
        else:
             beh_cpu = None

        for idx, p in enumerate(self.population):
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
            if idx < 5: # Debug first 5 agents
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
            if beh_cpu is not None:
                 b_row = beh_cpu[idx]
                 count = b_row[3]
                 if count > 0:
                     avg_speed = b_row[0] / count
                     avg_steer = b_row[1] / count
                     avg_dist = b_row[2] / count
                 else:
                     avg_speed = 0.0
                     avg_steer = 0.0
                     avg_dist = 0.0
            else:
                # Fallback to legacy buffer (should not be reached if tensor exists)
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
        # Replacement Logic
        with torch.no_grad():
            for idx in cull_indices:
                loser = self.population[idx]
                
                # Tournament Selection from candidates
                competitors = random.sample(parent_candidates, 3)
                competitors.sort(key=lambda i: (self.population[i]['rank'], -self.population[i]['crowding']))
                parent_idx = competitors[0]
                parent = self.population[parent_idx]
                
                # 1. Clone Params & Buffers (Stacked)
                for name, p_stack in self.stacked_params.items():
                    p_stack[idx].copy_(p_stack[parent_idx])
                for name, b_stack in self.stacked_buffers.items():
                    b_stack[idx].copy_(b_stack[parent_idx])
                
                # 2. Clone/Mutate Metadata
                loser['lr'] = parent['lr']
                loser['ent_coef'] = parent.get('ent_coef', self.config.ent_coef)
                
                # Clone EMAs
                loser['ema_efficiency'] = parent['ema_efficiency']
                loser['ema_wins'] = parent['ema_wins']
                loser['ema_consistency'] = parent['ema_consistency']
                loser['ema_laps'] = parent['ema_laps']
                loser['ema_runner_vel'] = parent['ema_runner_vel']
                loser['ema_blocker_dmg'] = parent['ema_blocker_dmg']
                loser['ema_behavior'] = parent['ema_behavior'].clone() if parent.get('ema_behavior') is not None else None
                
                # Mutate Rewards
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
                    
                # 3. Update Vectorized Optimizer State
                self.vectorized_adam.lrs[idx] = loser['lr']
                
                for name, m_stack in self.vectorized_adam.exp_avg.items():
                     m_stack[idx].copy_(m_stack[parent_idx])
                for name, v_stack in self.vectorized_adam.exp_avg_sq.items():
                     v_stack[idx].copy_(v_stack[parent_idx])
                
                if random.random() < 0.3:
                    loser['ent_coef'] *= random.uniform(0.8, 1.2)
                    loser['ent_coef'] = max(0.0001, min(0.1, loser['ent_coef']))
                if random.random() < 0.3:
                    loser['clip_range'] *= random.uniform(0.8, 1.2)
                    loser['clip_range'] = max(0.05, min(0.4, loser['clip_range']))

                self.log(f"Agent {loser['id']} (Rank {loser['rank']}) replaced by clone of {parent['id']} (Rank {parent['rank']})")
                
                # Update Global Reward Tensor
                start_idx = loser['id'] * self.config.envs_per_agent
                end_idx = start_idx + self.config.envs_per_agent
                for k, v in new_weights.items():
                    self.reward_weights_tensor[start_idx:end_idx, k] = v
        
        # Sync back to instances for saving/compatibility
        self.sync_vectorized_to_agents()
                 
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
        # NEW: Stage-based structure
        stage_dir = f"data/stage_{self.env.curriculum_stage}"
        gen_dir = os.path.join(stage_dir, f"gen_{self.generation}")
        
        os.makedirs(gen_dir, exist_ok=True)
        self.log(f"Saving generation {self.generation} to {gen_dir}...")
        
        # Identify Top 2 for League (Consistent with evolve logic)
        sorted_pop = sorted(self.population, key=lambda x: (x['laps_score'], x['checkpoints_score'], x['reward_score']), reverse=True)
        top_2 = sorted_pop[:2]
        
        for p in self.population:
            agent_id = p['id']
            save_path = os.path.join(gen_dir, f"agent_{agent_id}.pt")
            
            # Clean state_dict (Remove torch.compile _orig_mod prefix)
            state_dict = p['agent'].state_dict()
            if state_dict and next(iter(state_dict.keys())).startswith("_orig_mod."):
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                
            torch.save(state_dict, save_path)
            
            # Register Top Agents to League
            # START SOTA UPDATE: Enforce strict entry criteria
            # Only save to League if passing the "Team" bar (Stage 4+ and Diff > 0.5)
            # User Constraint: "we don't wan't to save agent in the league before they reached stage 4 (team)"
            if self.env.curriculum_stage >= STAGE_TEAM and self.env.bot_difficulty > 0.6:
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

        # --- Auto-Flush Old Generations (Stage-Specific, Date-Based) ---
        try:
            MAX_KEEP = self.config.max_checkpoints_to_keep
            
            # Scan current stage directory
            if os.path.exists(stage_dir):
                all_gens = []
                for d in os.listdir(stage_dir):
                    path = os.path.join(stage_dir, d)
                    if d.startswith("gen_") and os.path.isdir(path):
                        # Use modification time for pruning
                        mtime = os.path.getmtime(path)
                        all_gens.append((mtime, path, d))
            
                # Sort by Time (Oldest First)
                all_gens.sort(key=lambda x: x[0]) 
                
                # Prune
                if len(all_gens) > MAX_KEEP:
                    to_delete = all_gens[:-MAX_KEEP]
                    for _, path, d_name in to_delete:
                        shutil.rmtree(path)
                        self.log(f"Auto-Flush: Deleted old generation {d_name} from {stage_dir} (Limit {MAX_KEEP})")
                    
        except Exception as e:
            self.log(f"Auto-Flush Warning: {e}")
    def log_iteration_summary(self, global_step, sps, current_tau, avg_loss, avg_pg=0.0, avg_v=0.0, avg_ent=0.0, avg_div=0.0):
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
        self.log(f" Stage: {self.env.curriculum_stage} | Difficulty: {self.env.bot_difficulty:.2f} | Tau: {current_tau:.2f} | Spirit: {self.team_spirit:.2f} | Step Pen: {curr_step_pen:.1f}")
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


        self.log(f" Loss: {avg_loss:.4f} | PG: {avg_pg:.4f} | Val: {avg_v:.4f} | Ent: {avg_ent:.4f} | Div: {avg_div:.4f}")
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
        sps = 0.0 # Initialize to avoid unbound error on first steps
        
        self.last_transition_iteration = -999 # Initialize
        
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

                           
                           
                       # [FIX] Simplified Mitosis: Just Clone. 
                       # Diagnostics showed that Zeroing Teammate Weights caused 'Reverse' (-0.99) behavior 
                       # compared to preserving them (+0.86), likely due to learned bias from Stage 2 padding (-100k).
                       # We rely on training to adapt the weights to the new input distribution.
                               


                      # [FIX] CRITICAL: Sync Modified Agents back to Vectorized Stack!
                      # Without this, the optimizer continues using random Blocker weights.
                      self.log(">>> [FIX] Syncing Mitosis changes to Vectorized Parameters... <<<")
                      self.sync_agents_to_vectorized()
                      
                      # Reset Vectorized Optimizer State (Critical to break old momentum)
                      self.log(">>> [MITOSIS] Resetting Vectorized Optimizer State to clear Stage 2 momentum. <<<")
                      self.vectorized_adam.reset_state()
                      
                 
                 # GENERATE NEW MATCH ID (Forces Frontend Reset)
                 self.match_id = str(uuid.uuid4())
                 
                 obs_data = self.env.get_obs()


            # --- Dynamic Config Check ---
            current_stage = self.env.curriculum_stage
            
            # SOTA Tuning (See stage_0_tuning_report.md)
            # Nursery: Fast Evolution (1) to find movers. Steps 256.
            # Solo+: Stable Evolution (2). Steps 256.
            
            # SOTA Tuning (See stage_0_tuning_report.md)
            # Delegated to Stage Class (Evolve Interval)
            # target_steps removed - fixed to config.num_steps (512)
            target_evolve = self.curriculum.current_stage.target_evolve_interval
            
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

            # Re-allocate buffers (Always, as they are consumed by update loop to save VRAM)
            self.allocate_buffers()
                 
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
            
            # CRITICAL: Disable Autograd for entire collection phase to prevent graph growth
            _prev_grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
            
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
                
                t_loop_start = time.time() # T1

                # === VECTORIZED INFERENCE ===
                Pop = self.config.pop_size
                N_A = len(active_pods)
                EPA = self.config.envs_per_agent
                
                # Reshape for VMAP
                # Source: [N_Active, TotalEnvs, D] (norm_self is still flat [N*T, D] until reshaped)
                # Wait, code above line 1152 defined norm_self.view(...)
                # But here we are replacing starting at 1152.
                # Previous logic: norm_self = self.rms_self(raw_self) -> [N*T, D].
                
                v_s = norm_self.view(N_A, Pop, EPA, 15).permute(1, 0, 2, 3).reshape(Pop, -1, 15)
                v_tm = norm_tm.view(N_A, Pop, EPA, 13).permute(1, 0, 2, 3).reshape(Pop, -1, 13)
                v_en = norm_en.view(N_A, Pop, EPA, 2, 13).permute(1, 0, 2, 3, 4).reshape(Pop, -1, 2, 13)
                v_cp = norm_cp.view(N_A, Pop, EPA, 6).permute(1, 0, 2, 3).reshape(Pop, -1, 6)

                # Initialize Current States if first step
                if step == 0:
                     # [Pop, Batch, 1, Hidden]
                     total_batch = v_s.shape[1]
                     self.curr_actor_h = torch.zeros((Pop, total_batch, 1, self.config.lstm_hidden_size), device=self.device)
                     self.curr_actor_c = torch.zeros((Pop, total_batch, 1, self.config.lstm_hidden_size), device=self.device)
                     self.curr_critic_h = torch.zeros((Pop, total_batch, 1, self.config.lstm_hidden_size), device=self.device)
                     self.curr_critic_c = torch.zeros((Pop, total_batch, 1, self.config.lstm_hidden_size), device=self.device)
                
                # Check Sparse Store (Before Forward? No, we need state used for THIS step)
                if step % self.config.seq_length == 0:
                     chunk_idx = step // self.config.seq_length
                     for i in range(Pop):
                         batch = self.agent_batches[i]
                         batch['actor_h'][chunk_idx] = self.curr_actor_h[i].detach()
                         batch['actor_c'][chunk_idx] = self.curr_actor_c[i].detach()
                         batch['critic_h'][chunk_idx] = self.curr_critic_h[i].detach()
                         batch['critic_c'][chunk_idx] = self.curr_critic_c[i].detach()

                # VMAP Execution
                # Pass Current States
                # vmap expects inputs to be stacked. self.curr_... is [Pop, Batch, ...]
                
                # Add critic startes to input
                v_act, v_lp, _, v_val, (v_nah, v_nac), (v_nch, v_ncc) = vmap(self._functional_inference, in_dims=(0,0,0,0,0,0,0,0,0,0), randomness='different')(
                    self.stacked_params, self.stacked_buffers, v_s, v_tm, v_en, v_cp, self.curr_actor_h, self.curr_actor_c, self.curr_critic_h, self.curr_critic_c
                )
                
                # Update Current States
                self.curr_actor_h = v_nah
                self.curr_actor_c = v_nac
                # Critic states are not returned by inference?
                # Wait, I updated _functional_inference to only return actor states in previous tool call?
                # "Output: action, logprob, ent, val, states"
                # And functional_call returns what PodAgent returns.
                # PodAgent.forward returns: ..., (ah, ac), (ch, cc)
                # So yes, it returns BOTH.
                self.curr_critic_h = v_nch
                self.curr_critic_c = v_ncc
                
                # Fix vmap shape artifacts (Batch x Batch) if present
                
                # Helper for diagonal fix on [P, B, B, ...]
                def fix_diag(t):
                    # 1. Check for 5D States [P, L, B, B, H] - Artifact at dim 2 and 3
                    if t.ndim == 5 and t.shape[2] == t.shape[3]:
                        return t.diagonal(dim1=2, dim2=3).permute(0, 3, 1, 2).contiguous()

                    # 2. Check for 3D/4D Actions/LogProbs - Artifact at dim 1 and 2
                    if t.ndim >= 3 and t.shape[1] == t.shape[2]:
                        # [P, B, B, ...] -> [P, ..., B]
                        # diagonal(dim1=1, dim2=2) moves diag to FASTEST dimension (last).
                        # Input [P, B, B, F]. Output [P, F, B].
                        # We need [P, B, F].
                        # So permute(-1) to dim 1.
                        # If t is [P, B, B, F]. Result [P, F, B]. Permute(0, 2, 1).
                        
                        if t.ndim == 4: # Act [P, B, B, F]
                            return t.diagonal(dim1=1, dim2=2).permute(0, 2, 1).contiguous()
                        elif t.ndim == 3: # LogP [P, B, B]
                            return t.diagonal(dim1=1, dim2=2) # [P, B]
                    
                    # 3. Canonicalize 4D State Output [P, L, B, H] -> [P, B, L, H]
                    if t.ndim == 4 and t.shape[1] == 1 and t.shape[2] != 1:
                         return t.permute(0, 2, 1, 3).contiguous()

                    return t

                v_act = fix_diag(v_act)
                v_lp = fix_diag(v_lp)
                v_val = fix_diag(v_val)
                
                v_nah = fix_diag(v_nah)
                v_nac = fix_diag(v_nac)
                v_nch = fix_diag(v_nch)
                v_ncc = fix_diag(v_ncc)

                # Update current states
                self.curr_actor_h = v_nah
                self.curr_actor_c = v_nac
                self.curr_critic_h = v_nch
                self.curr_critic_c = v_ncc
                
                # Store
                for i in range(Pop):
                    batch = self.agent_batches[i]
                    batch['self_obs'][step] = v_s[i].detach()
                    batch['teammate_obs'][step] = v_tm[i].detach()
                    batch['enemy_obs'][step] = v_en[i].detach()
                    batch['cp_obs'][step] = v_cp[i].detach()
                    
                    batch['actions'][step] = v_act[i].detach().reshape(self.config.envs_per_agent * 1, 4)
                    batch['logprobs'][step] = v_lp[i].detach().flatten()
                    batch['values'][step] = v_val[i].detach().flatten()
                    
                # Prepare act_map
                r_act = v_act.view(Pop, N_A, EPA, 4)
                out_act = r_act.permute(1, 0, 2, 3).reshape(N_A, Pop * EPA, 4)
                
                act_map = {}
                for idx, pid in enumerate(active_pods):
                     act_map[pid] = out_act[idx]
                
                # --- Timing Debug ---
                t_infer_end = time.time() # T2
                
                # Check for League Mode Opponent Logic (Stage 2: 2v2)
                # ... Excluded ...
                
                # ... Excluded Opponent Logic Reuse ...
                # (Assuming no change needed there for now as opponents need states too? 
                # Opponent Agent uses MLP or LSTM?
                # If Opponent is Loaded from old checkpoint, it might be MLP.
                # If loaded from New Gen, it is LSTM.
                # We need to handle this polymorphism in league logic!)
                
                # For now, let's assume we focus on Training Loop logic.
                # League logic block was skipped in replacement to keep it simple? 
                # No, I need to preserve it if I replace large chunk.
                # I am replacing lines 1282-1416. 
                # This covers VMAP Execution and Store. 
                # It ENDS before League Logic.

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
                
                t_inf_end = time.time() # This is actually T2 (Infer End) in logic, but here it's before Step.
                # Let's rename for clarity:
                t_pre_step = time.time() # T3
                with torch.no_grad():
                     rewards_all, dones, infos = self.env.step(env_actions, reward_weights=self.reward_weights_tensor, tau=current_tau, team_spirit=self.team_spirit)
                t_step_end = time.time() # T4
                
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

                t_post_reward = time.time() # T5
                
                # --- Per-Agent Stats Loop (VECTORIZED ACCUMULATION) ---
                # CRITICAL OPTIMIZATION: Removed .item() calls from inner loop.
                # We now accumulate tensors and reduce ONCE at the end of iteration.
                
                # Checkpoints
                # Filter infos first
                # infos[...] are [4096, 4] -> [Pop*Envs, 4]
                
                # We need to sum per agent for the whole batch.
                # Since agents are contiguous in dim 0 (rows), we can reshape and sum.
                # Shape: [Pop, EnvsPerAgent, 4]
                
                # Metrics to track:
                # 0. Reward Sum (Raw)
                # 1. Laps
                # 2. Checkpoints
                # 3. Runner Vel
                # 4. Blocker Dmg
                # 5. Max Streak (Max is tricky to vector-accumulate without expansive memory, let's keep max per step?)
                # 6. Efficiency (Steps/Hits)
                # 7. Wins
                # 8. Matches
                
                # 0. Rewards Raw
                # rewards_all: [TotalEnvs, 4]
                # To get per-agent mean:
                # Reshape [Pop, EnvsPerAgent, 4]
                r_reshaped = rewards_all.view(self.config.pop_size, self.config.envs_per_agent, 4)
                
                # Filter by active pods
                # active_pods is list. 
                # r_active = r_reshaped[:, :, active_pods] -> [Pop, EnvsPerAgent, N_Active]
                r_active = r_reshaped[:, :, active_pods]
                
                # Sum over Envs and Pods, then divide by Count later
                # We accumulate SUM and COUNT in a tensor buffer on GPU.
                # We need a buffer for this iteration.
                # Let's use a temporary buffer dict initialized outside loop?
                # Or just accumulate into a 'metrics_buffer' tensor on self used for this loop?
                
                if not hasattr(self, '_iter_metrics'):
                     # Allocate ONCE
                     # [Pop, 10]
                     # 0: RewardSum, 1: RewardCount, 2: Laps, 3: CPs, 4: Vel, 5: Dmg, 
                     # 6: CP_Steps, 7: CP_Hits, 8: Wins, 9: Matches
                     self._iter_metrics = torch.zeros((self.config.pop_size, 10), device=self.device)
                     self._iter_max_streak = torch.zeros((self.config.pop_size,), device=self.device) # Keep max separate
                
                # 0. Rewards
                # Sum over (Envs, ActivePods)
                r_sum = r_active.sum(dim=(1, 2)) # [Pop]
                r_count = r_active.numel() / self.config.pop_size # Scalar constant (EnvsPerAgent * N_Active)
                
                self._iter_metrics[:, 0] += r_sum
                self._iter_metrics[:, 1] += r_count
                
                # 1 & 2. Laps & CPs
                # infos['laps_completed']: [TotalEnvs, 4]
                i_laps = infos['laps_completed'].view(self.config.pop_size, self.config.envs_per_agent, 4)
                i_cps = infos['checkpoints_passed'].view(self.config.pop_size, self.config.envs_per_agent, 4)
                i_streak = infos['current_streak'].view(self.config.pop_size, self.config.envs_per_agent, 4)
                
                # Filter Active
                l_active = i_laps[:, :, active_pods]
                # CP Logic: passed > 0 AND streak > 0
                c_passed = (i_cps[:, :, active_pods] > 0)
                c_streak = (i_streak[:, :, active_pods] > 0)
                c_valid = (c_passed & c_streak).float()
                
                self._iter_metrics[:, 2] += l_active.sum(dim=(1, 2))
                self._iter_metrics[:, 3] += c_valid.sum(dim=(1, 2))
                
                # 3 & 4. Roles
                i_vel = infos['runner_velocity'].view(self.config.pop_size, self.config.envs_per_agent, 4)
                i_dmg = infos['blocker_damage'].view(self.config.pop_size, self.config.envs_per_agent, 4)
                
                self._iter_metrics[:, 4] += i_vel[:, :, active_pods].sum(dim=(1, 2))
                self._iter_metrics[:, 5] += i_dmg[:, :, active_pods].sum(dim=(1, 2))
                
                # 5. Max Streak
                # Max over (Envs, ActivePods)
                # i_streak_active = i_streak[:, :, active_pods]
                # current_iter_max = i_streak_active.flatten(1).max(dim=1).values # [Pop]
                # self._iter_max_streak = torch.max(self._iter_max_streak, current_iter_max)
                
                # Optimized Max:
                # Only check active pods
                streak_active = i_streak[:, :, active_pods]
                # Max across Envs and Pods per agent
                batch_max = streak_active.amax(dim=(1, 2)) # [Pop]
                self._iter_max_streak = torch.maximum(self._iter_max_streak, batch_max)

                # 6. Efficiency
                # Filter CP1 Farmers (streak > 0)
                i_steps = infos['cp_steps'].view(self.config.pop_size, self.config.envs_per_agent, 4)[:, :, active_pods]
                
                # Mask: steps > 0 AND streak > 0
                # We need corresponding streak for active pods
                s_active = i_streak[:, :, active_pods]
                
                eff_mask = (i_steps > 0) & (s_active > 0)
                
                # Sum Steps where mask is true
                # We can just multiply steps * mask.float()
                eff_steps = (i_steps * eff_mask.float()).sum(dim=(1, 2))
                eff_hits = eff_mask.float().sum(dim=(1, 2))
                
                self._iter_metrics[:, 6] += eff_steps
                self._iter_metrics[:, 7] += eff_hits
                
                # 7 & 8. Wins & Matches
                # Dones [TotalEnvs] -> [Pop, EnvsPerAgent]
                d_reshaped = dones.view(self.config.pop_size, self.config.envs_per_agent)
                
                # Winners [TotalEnvs] -> [Pop, EnvsPerAgent]
                w_reshaped = self.env.winners.view(self.config.pop_size, self.config.envs_per_agent)
                
                # Win if winner == 0 (Pod 0 check is specific to "Me vs Enemy" setup where Me is Team 0)
                # env.winners is Team ID (0 or 1). 
                # We assume Agent is always Team 0 in its Environment View?
                # Yes, env.py usually sets up 'self' as Team 0 relative to camera/obs, 
                # but 'winners' is absolute Team ID.
                # In Solo/Duel, Agent controls local Team 0.
                
                # Mask where Done is True
                done_mask = d_reshaped.bool()
                
                # Wins where done AND winner == 0
                wins = (done_mask & (w_reshaped == 0)).float().sum(dim=1)
                matches = done_mask.float().sum(dim=1)
                
                self._iter_metrics[:, 8] += wins
                self._iter_metrics[:, 9] += matches
                
                
                # --- Fill Batch Data (Vectorized) ---
                for i in range(self.config.pop_size):
                    # We still need to fill the batch buffers for PPO Update
                    start_env = i * self.config.envs_per_agent
                    end_env = start_env + self.config.envs_per_agent
                    
                    # We can't fully vectorize this assignment without changing self.agent_batches structure to a big tensor.
                    # But assignment slice is fast enough (metadata op).
                    # The slow part was the scalar accumulation logic above.
                    
                    # Store Sliced Tensors (Already on GPU)
                    r_slice = norm_rewards[start_env:end_env, active_pods].flatten()
                    d_slice = dones[start_env:end_env].repeat_interleave(len(active_pods))
                    
                    self.agent_batches[i]['rewards'][step] = r_slice
                    self.agent_batches[i]['dones'][step] = d_slice
                    
                t_post_stats = time.time() # T6
                    
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
                
                t_post_telemetry = time.time() # T7
                
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
                
                # Vectorized Accumulation to avoid 128x Python Loop Syncs
                # sum_speed: [Pop * EnvsPerAgent]
                # sum_steer: [Pop * EnvsPerAgent]
                # sum_dist:  [Pop * EnvsPerAgent]
                
                # Reshape and Sum per Agent
                # [Pop, EnvsPerAgent] -> [Pop]
                ag_speed = sum_speed.view(self.config.pop_size, self.config.envs_per_agent).sum(dim=1)
                ag_steer = sum_steer.view(self.config.pop_size, self.config.envs_per_agent).sum(dim=1)
                ag_dist  = sum_dist.view( self.config.pop_size, self.config.envs_per_agent).sum(dim=1)
                
                # Count: EnvsPerAgent * NumActivePods
                count_val = self.config.envs_per_agent * len(active_pods)
                
                if hasattr(self, 'behavior_stats_tensor'):
                     self.behavior_stats_tensor[:, 0] += ag_speed
                     self.behavior_stats_tensor[:, 1] += ag_steer
                     self.behavior_stats_tensor[:, 2] += ag_dist
                     self.behavior_stats_tensor[:, 3] += count_val
                
                # Legacy legacy
                pass
                     
                t_behavior_end = time.time() # T8 Ends Behavior

                # Manual Reset
                if dones.any():
                    # Valid Dones
                    done_mask = dones.view(self.config.num_envs, 1, 1, 1) # [Envs, 1, 1, 1]
                    # We need to map dones to the vectorized state [Pop, Batch, 1, H]
                    # Batch dim in vmap is [N_Active, Envs/Agent].
                    # Wait, Batch size is Pop * EnvsPerAgent. vmap input is [Pop, BatchPerAgent].
                    # dones is [TotalEnvs].
                    # reshaping dones to [Pop, BatchPerAgent, 1, 1]
                    
                    # Dones is [Pop * EnvsPerAgent]
                    d_reshaped = dones.view(Pop, EPA).unsqueeze(-1).unsqueeze(-1) # [Pop, EPA, 1, 1]
                    # Expand to Active Pods dimension?
                    # The state tensor is [Pop, N_Active * EPA, 1, H]
                    # We need to repeat dones for each active pod.
                    d_expanded = d_reshaped.repeat(1, N_A, 1, 1) # [Pop, N_A * EPA, 1, 1]
                    # Reshape to match state [Pop, Batch, 1, 1]
                    d_mask = d_expanded.reshape(Pop, -1, 1, 1)
                    
                    # Apply Mask (Reset state to zero if done)
                    self.curr_actor_h = self.curr_actor_h * (1.0 - d_mask.float())
                    self.curr_actor_c = self.curr_actor_c * (1.0 - d_mask.float())
                    self.curr_critic_h = self.curr_critic_h * (1.0 - d_mask.float())
                    self.curr_critic_c = self.curr_critic_c * (1.0 - d_mask.float())

                    reset_ids = torch.nonzero(dones).squeeze(-1)
                    self.env.reset(reset_ids)
                
                t_reset_end = time.time() # T9 Ends Reset

                # Next Obs (New Start State)
                with torch.no_grad():
                    obs_data = self.env.get_obs()
                t_obs_end = time.time() # T10 Ends Obs
                
                if step % 100 == 0:
                     if os.environ.get("ENABLE_PROFILING"):
                        self.log(f"Profile (Step {step}): A_Inf={(t_infer_end-t_loop_start)*1000:.1f} | B_Leag={(t_pre_step-t_infer_end)*1000:.1f} | C_Step={(t_step_end-t_pre_step)*1000:.1f} | D_Rew={(t_post_reward-t_step_end)*1000:.1f} | E_Stat={(t_post_stats-t_post_reward)*1000:.1f} | F_Tel={(t_post_telemetry-t_post_stats)*1000:.1f} | G_Beh={(t_behavior_end-t_post_telemetry)*1000:.1f} | H_Res={(t_reset_end-t_behavior_end)*1000:.1f} | I_Obs={(t_obs_end-t_reset_end)*1000:.1f}")
                    
                all_self, all_tm, all_en, all_cp = obs_data

            # Restore Grad
            torch.set_grad_enabled(_prev_grad_state)

            # --- POST-COLLECTION REDUCTION (SYNC CPU ONCE) ---
            # Flush metrics from GPU to CPU
            if hasattr(self, '_iter_metrics'):
                # Copy to CPU
                metrics_cpu = self._iter_metrics.cpu().numpy()
                max_streak_cpu = self._iter_max_streak.cpu().numpy()
                
                # 0: RewardSum, 1: RewardCount, 2: Laps, 3: CPs, 4: Vel, 5: Dmg, 
                # 6: CP_Steps, 7: CP_Hits, 8: Wins, 9: Matches
                
                for i in range(self.config.pop_size):
                    m = metrics_cpu[i]
                    
                    if m[1] > 0:
                        self.population[i]['reward_score'] += (m[0] / m[1])
                    
                    self.population[i]['laps_score'] += int(m[2])
                    self.population[i]['checkpoints_score'] += int(m[3])
                    self.population[i]['avg_runner_vel'] += m[4]
                    self.population[i]['avg_blocker_dmg'] += m[5]
                    
                    # Logic needs to account for max streak being tracked per step? 
                    # Yes, we did max over batch per step accumulator.
                    cur_max = int(max_streak_cpu[i])
                    if cur_max > self.population[i]['max_streak']:
                        self.population[i]['max_streak'] = cur_max
                        
                    self.population[i]['total_cp_steps'] += int(m[6])
                    self.population[i]['total_cp_hits'] += int(m[7])
                    
                    self.population[i]['wins'] += int(m[8])
                    self.population[i]['matches'] += int(m[9])
                
                # Reset Buffers
                self._iter_metrics.zero_()
                self._iter_max_streak.zero_()

            # 3. Update Phase (Per Agent)
            # Bootstrapping & Training
            total_loss = 0
            
            t_gae_start = time.time()
            
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
            # === VECTORIZED UPDATE PHASE ===
            bs_per_agent = self.config.envs_per_agent * len(active_pods)
            Pop = self.config.pop_size
            N_A = len(active_pods)
            EPA = self.config.envs_per_agent

            # 1. Next Values (Vectorized)
            # Reshape next_norm_* [N_A, TotalEnvs, D] -> [Pop, Batch, D]
            v_n_s = next_norm_self.view(N_A, Pop, EPA, 15).permute(1, 0, 2, 3).reshape(Pop, -1, 15)
            v_n_tm = next_norm_tm.view(N_A, Pop, EPA, 13).permute(1, 0, 2, 3).reshape(Pop, -1, 13)
            v_n_en = next_norm_en.view(N_A, Pop, EPA, 2, 13).permute(1, 0, 2, 3, 4).reshape(Pop, -1, 2, 13)
            v_n_cp = next_norm_cp.view(N_A, Pop, EPA, 6).permute(1, 0, 2, 3).reshape(Pop, -1, 6)
            
            with torch.no_grad():
                # Use current critic states (at end of rollout)
                v_next_vals = vmap(self._functional_get_value, in_dims=(0,0,0,0,0,0,0,0), randomness='different')(
                     self.stacked_params, self.stacked_buffers, v_n_s, v_n_tm, v_n_en, v_n_cp, self.curr_critic_h, self.curr_critic_c
                )
            
            # Flatten to [TotalBatch] matching g_dones structure
            g_next_val = v_next_vals.flatten()
            
            # 2. Global GAE Construction (On Full Trajectories FIRST)
            # Stack agent batches [Pop, Time, BPA]
            # Use pop() to clear from agent_batches immediately to save VRAM
            # We pop STATES too
            s_rew = torch.stack([b.pop('rewards') for b in self.agent_batches]) 
            s_don = torch.stack([b.pop('dones') for b in self.agent_batches])
            s_val = torch.stack([b.pop('values') for b in self.agent_batches])
            
            # Transpose to [Time, Pop, BPA] -> Flatten to [Time, TotalBatch]
            g_rewards = s_rew.permute(1, 0, 2).reshape(self.current_num_steps, -1)
            g_dones = s_don.permute(1, 0, 2).reshape(self.current_num_steps, -1)
            g_values = s_val.permute(1, 0, 2).reshape(self.current_num_steps, -1)
            
            # 3. GAE Loop (Standard Full Trajectory)
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

            # Free GAE intermediates
            del s_rew, s_don, g_rewards, g_dones, g_values, g_next_val


            # 4. SEQUENCE CHUNKING & FLATTENING (TBPTT Prep)
            t_ppo_start = time.time()

            # Helper to chunk and flatten: [Pop, Steps, BPA, D] -> [Pop, TotalSeqs, SeqLen, D]
            SeqLen = self.config.seq_length
            NumChunks = self.current_num_steps // SeqLen

            def transform_seq(tensor_stack):
                # tensor_stack: [Pop, Steps, BPA, ...]
                # View as [Pop, Chunks, SeqLen, BPA, ...]
                # Permute to [Pop, Chunks, BPA, SeqLen, ...]
                # Flatten [Chunks, BPA] -> TotalSeqs
                shape = tensor_stack.shape
                new_shape = (Pop, NumChunks, SeqLen, -1) + shape[3:]
                viewed = tensor_stack.view(*new_shape)
                # Permute: [Pop, Chunks, BPA, SeqLen, ...]
                # Original dims: 0=Pop, 1=Chunks, 2=SeqLen, 3=BPA
                permuted = viewed.permute(0, 1, 3, 2, *range(4, viewed.ndim))
                # Flatten Chunks*BPA
                return permuted.flatten(1, 2) # [Pop, TotalSeqs, SeqLen, ...]

            # Helper for States: [Pop, Chunks+1, BPA, ...] -> [Pop, TotalSeqs, ...]
            def transform_state(tensor_stack):
                # tensor_stack: [Pop, Chunks+1, BPA, L, H]
                # Take only first NumChunks
                valid = tensor_stack[:, :NumChunks]
                # Flatten [Chunks, BPA] -> TotalSeqs
                return valid.flatten(1, 2) # [Pop, TotalSeqs, L, H]

            # Transformation
            # Stack Data: [Pop, Steps, BPA, D]
            flat_s = transform_seq(torch.stack([b.pop('self_obs') for b in self.agent_batches]))
            flat_tm = transform_seq(torch.stack([b.pop('teammate_obs') for b in self.agent_batches]))
            flat_en = transform_seq(torch.stack([b.pop('enemy_obs') for b in self.agent_batches]))
            flat_cp = transform_seq(torch.stack([b.pop('cp_obs') for b in self.agent_batches]))
            flat_act = transform_seq(torch.stack([b.pop('actions') for b in self.agent_batches]))
            flat_old_logp = transform_seq(torch.stack([b.pop('logprobs') for b in self.agent_batches]))
            
            # transform_seq works on s_val reference too? No, s_val was consumed/deleted.
            # We need to reshape G_ADV and G_RET.
            # g_adv is [Steps, TotalBatch]. TotalBatch = Pop * BPA.
            # We need [Pop, Steps, BPA].
            g_adv_reshaped = g_adv.view(self.current_num_steps, Pop, -1).permute(1, 0, 2)
            g_ret_reshaped = g_returns.view(self.current_num_steps, Pop, -1).permute(1, 0, 2)
            # Reconstruct Old Value (Optional?) - No, we popped it. 
            # But we can reconstruct logic or store? 
            # Actually we typically need Old Value for Clip Range.
            # Let's assume we don't have it unless we stored it in 'values' buffer which we popped.
            # Wait, s_val was popped on 1930. We can't reuse it.
            # We should have kept a copy or transformed it.
            # Since we deleted s_val, we rely on g_values from line 1935? 
            # g_values was also deleted on 1955!
            # CRITICAL MISS: We need old_values for PPO loss.
            # FIX: Do not delete g_values until transformed.
            # But g_values is [Steps, TotalBatch].
            
            # Let's assume we missed saving g_values. 
            # PPO Value Clip relies on it. 
            # Re-calculating...
            # We can recover Old Value from g_returns - g_adv?
            # Returns = Adv + Value -> Value = Returns - Adv.
            g_values_recovered = g_returns - g_adv # Exact reconstruction
            
            flat_old_val = transform_seq(g_values_recovered.view(self.current_num_steps, Pop, -1).permute(1, 0, 2).unsqueeze(-1)).squeeze(-1)
            flat_adv = transform_seq(g_adv_reshaped.unsqueeze(-1)).squeeze(-1)
            flat_ret = transform_seq(g_ret_reshaped.unsqueeze(-1)).squeeze(-1)
            
            # Transform States
            flat_ah = transform_state(torch.stack([b.pop('actor_h') for b in self.agent_batches]))
            flat_ac = transform_state(torch.stack([b.pop('actor_c') for b in self.agent_batches]))
            flat_ch = transform_state(torch.stack([b.pop('critic_h') for b in self.agent_batches]))
            flat_cc = transform_state(torch.stack([b.pop('critic_c') for b in self.agent_batches]))

            # Config Tensors
            t_ent = torch.tensor([p.get('ent_coef', self.config.ent_coef) for p in self.population], device=self.device)
            t_clip = torch.tensor([p.get('clip_range', self.config.clip_range) for p in self.population], device=self.device)
            t_vf = torch.tensor(self.config.vf_coef, device=self.device)
            t_div_coef = torch.tensor(self.config.div_coef, device=self.device)
            use_div = (self.env.curriculum_stage >= STAGE_TEAM)
            
            # Training Loop (Sequence Batches)
            num_sequences = flat_s.shape[1] # TotalSeqs
            inds = np.arange(num_sequences)
            
            # Minibatch Size (Sequences)
            minibatch_size = max(1, num_sequences // self.config.num_minibatches)
            
            # Compiled Grad Function
            compute_grad = vmap(grad(self._functional_loss, has_aux=True), in_dims=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, None,0,0,None,None))
            
            total_loss = 0.0
            total_pg = 0.0
            total_v = 0.0
            total_ent = 0.0
            total_div = 0.0
            
            for _ in range(self.config.update_epochs):
                 np.random.shuffle(inds)
                 for start in range(0, num_sequences, minibatch_size):
                      end = start + minibatch_size
                      mb_inds = inds[start:end]
                      
                      # Slice [Pop, MB, Seq, D]
                      m_s = flat_s[:, mb_inds]
                      m_tm = flat_tm[:, mb_inds]
                      m_en = flat_en[:, mb_inds]
                      m_cp = flat_cp[:, mb_inds]
                      m_act = flat_act[:, mb_inds]
                      m_lp = flat_old_logp[:, mb_inds]
                      m_v = flat_old_val[:, mb_inds]
                      m_adv = flat_adv[:, mb_inds]
                      m_ret = flat_ret[:, mb_inds]
                      
                      # States [Pop, MB, L, H] (Initial state for chunk)
                      m_ah = flat_ah[:, mb_inds]
                      m_ac = flat_ac[:, mb_inds]
                      m_ch = flat_ch[:, mb_inds]
                      m_cc = flat_cc[:, mb_inds]
                      
                      # Compute Vectorized Gradients
                      grads, batch_aux = compute_grad(
                          self.stacked_params, self.stacked_buffers,
                          m_s, m_tm, m_en, m_cp, m_act, m_lp, m_v, m_adv, m_ret, 
                          m_ah, m_ac, m_ch, m_cc,
                          use_div, t_ent, t_clip, t_vf, t_div_coef
                      )
                      
                      # Unpack Aux: Tuple of Tensors [Pop, MB]
                      b_loss, b_pg, b_v, b_ent, b_div, *_ = batch_aux
                      
                      total_loss += b_loss.sum().item()
                      total_pg += b_pg.sum().item()
                      total_v += b_v.sum().item()
                      total_ent += b_ent.sum().item()
                      total_div += b_div.sum().item()
                      
                      # Optimizer Step
                      self.vectorized_adam.step(grads)
                      
                      # RND Update (Flatten Sequences too)
                      # RND expects [TotalSamples, 15]
                      self.rnd.update(m_s.reshape(-1, 15).detach())

            # Stats Normalization
            num_updates = self.config.pop_size * self.config.update_epochs * (num_sequences // minibatch_size)
            
            avg_loss = total_loss / num_updates
            avg_pg = total_pg / num_updates
            avg_v = total_v / num_updates
            avg_ent = total_ent / num_updates
            avg_div = total_div / num_updates
            
            self.log_iteration_summary(global_step, sps, current_tau, avg_loss, avg_pg, avg_v, avg_ent, avg_div)
            
            # Construct simple line for telemetry log/frontend if needed (backward compat)
            leader = self.population[self.leader_idx]
            l_eff = leader.get('ema_efficiency')
            if l_eff is None: l_eff = 0.0
            log_line = f"Step: {global_step} | SPS: {sps} | Gen: {self.generation} | Leader Eff: {l_eff:.1f}"
            # self.log(log_line) # Suppressed to avoid double printing
            
            if telemetry_callback:
                # Pass league_stats as a new argument
                league_stats = None
                telemetry_callback(global_step, sps, 0, 0, self.current_win_rate, self.telemetry_env_indices[0], total_loss/self.config.pop_size, log_line, False, league_stats=league_stats)

            # --- Memory Optimization ---
            # Explicitly clear cache after heavy update loop
            import gc
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # --- Loop Progression & Evolution ---
            global_step += self.current_num_steps
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                sps = global_step / elapsed
            
            # Evolution Step
            if self.iteration % self.current_evolve_interval == 0:
                 self.log(f"Evolution Triggered at Iteration {self.iteration} (Interval: {self.current_evolve_interval})")
                 self.evolve_population()
                 # Reset optimizer state if needed? Usually PPO optimizer state is preserved or reset by design in evolve().
                 # PPO usually resets optimizer per generation if we use 'Genetic PPO' style where weights are mutated.
                 # If we use standard PPO, we don't mutate weights, just selection?
                 # evolve_population() logic decides.
            
            start_time_iter = time.time() # Reset iter timer if tracking per-iter SPS
            
            start_time = time.time()
            
            # Save Checkpoint (Leader)
            if self.iteration % 50 == 0:
                # Sync first!
                self.sync_vectorized_to_agents()
                # Save leader
                leader_agent = self.population[self.leader_idx]['agent']
                torch.save(leader_agent.state_dict(), f"data/checkpoints/model_gen{self.generation}_best.pt")

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train_loop()
