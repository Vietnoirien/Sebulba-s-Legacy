import torch
import math
from simulation.gpu_physics import GPUPhysics
from simulation.tracks import TrackGenerator
from config import *

# Reward Indices and Weights imported from config


class PodRacerEnv:
    def __init__(self, num_envs, device='cuda', start_stage=STAGE_SOLO):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.physics = GPUPhysics(num_envs, device=device)
        self.curriculum_stage = start_stage 
        
        # Default Config (Solo)
        self.config = EnvConfig(
             mode_name="solo",
             track_gen_type="max_entropy",
             active_pods=[0],
             use_bots=False,
             step_penalty_active_pods=[0],
             orientation_active_pods=[0]
        )
        
        # New Configs
        self.bot_config = BotConfig()
        self.spawn_config = SpawnConfig()
        self.reward_scaling_config = RewardScalingConfig()

        # Map Mode Cache
        self.using_nursery_map = False
        self._update_map_mode()
        
        # Game State
        self.next_cp_id = torch.ones((num_envs, 4), dtype=torch.long, device=self.device) # Start at 1 (0 is start)
        self.laps = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device)
        self.timeouts = torch.full((num_envs, 4), TIMEOUT_STEPS, dtype=torch.long, device=self.device)
        self.cps_passed = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device)
        
        # Checkpoints [Batch, MaxCP, 2]
        self.checkpoints = torch.zeros((num_envs, MAX_CHECKPOINTS, 2), device=self.device)
        self.num_checkpoints = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        
        # Track steps since last CP for efficiency metric
        self.steps_last_cp = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device) # Track steps per CP
        
        # Metrics per env
        self.stage_metrics = {
            "solo_completes": 0,
            "solo_steps": 0,
            "checkpoint_hits": 0,
            "duel_wins": 0,
            "duel_games": 0,
            "recent_wins": 0,
            "recent_games": 0,
            "recent_episodes": 0
        }
        
        # Rewards / Done
        self.rewards = torch.zeros((num_envs, 4), device=self.device) # Pod 0, 1, 2, 3
        self.dones = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self.winners = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        
        # Roles
        self.is_runner = torch.zeros((num_envs, 4), dtype=torch.bool, device=self.device)

        
        # Progress tracking for dense rewards
        self.prev_dist = torch.zeros((num_envs, 4), device=self.device)
        
        self.bot_difficulty = 0.0 # 0.0 to 1.0
        
        # Role Hysteresis
        self.role_lock_timer = torch.zeros((num_envs,), dtype=torch.long, device=self.device)


        
        # Rank Tracking for Potential-Based Reward
        self.prev_ranks = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device)
        self.rank_range_cache = torch.arange(4, device=self.device).unsqueeze(0).expand(num_envs, 4)
        
        self.reset()
        
    def _get_ranks(self, env_ids):
        """
        Calculate current rank (0-3) for each pod based on progress score.
        Score = Laps * 10000 + NextCP * 100 + (1 - Dist/20000)
        """
        # Gather data
        laps = self.laps[env_ids] # [N, 4]
        next_cp = self.next_cp_id[env_ids] # [N, 4]
        # Use prev_dist as it is updated every step/reset and represents 'dist to next cp'
        dists = self.prev_dist[env_ids] # [N, 4]
        
        # Max map dist approx 20000.
        dist_score = 1.0 - (dists / 20000.0)
        
        # FIX: Handle Final Leg (NextCP == 0)
        # NextCP=0 means passing last CP and aiming for Finish (Lap Completion).
        # We should count this as "NumCPs" progress, not 0.
        # Since NextCP starts at 1, 0 is unique to this state.
        effective_cp = next_cp.clone()
        # Create mask for envs where next_cp is 0
        mask_finish = (effective_cp == 0)
        
        # We need num_checkpoints for each env
        # self.num_checkpoints is [N_envs]
        # We need to broadcast or index correctly
        # env_ids is passed in, so we look up specific envs
        n_cps_local = self.num_checkpoints[env_ids]
        n_cps_expanded = n_cps_local.unsqueeze(1).expand(-1, 4)
        
        # Optimized: Use torch.where to avoid clone/mask assignment overhead
        # effective_cp = mask ? n_cps : next_cp
        effective_cp = torch.where(next_cp == 0, n_cps_expanded, next_cp)

        # Score calculation (Higher is better)
        # Using larger multipliers to ensure strict hierarchy: Lap > CP > Dist
        scores = (laps * 10000.0) + (effective_cp * 100.0) + dist_score
        
        # Sort scores descending to get ranks
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        
        # Creating rank tensor
        m_ranks = torch.zeros_like(scores, dtype=torch.long)
        
        # Optimized: Use pre-cached range if available, else allocate (fallback)
        if hasattr(self, 'rank_range_cache') and self.rank_range_cache.shape[0] == len(scores):
             rang = self.rank_range_cache
        else:
             rang = torch.arange(4, device=self.device).unsqueeze(0).expand(len(scores), 4)

        m_ranks.scatter_(1, sorted_indices, rang)
        
        return m_ranks

    def reset(self, env_ids=None):
        if env_ids is None:
            # All
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # 1. Generate Tracks (Vectorized Retry Logic)
        # Optimized: Fail fast (3 attempts, 20 loops) -> Fallback to Procedural Path
        
        # 1. Generate Tracks
        # Using Modular Generators for Diversity
        
        curr_env_ids = env_ids
        num_curr = len(curr_env_ids)
        
        if num_curr > 0:
            # 1. Reset CPs 
            self.checkpoints[curr_env_ids] = 0.0
            
            # 2. Random N CPs
            n_cps = torch.randint(MIN_CHECKPOINTS, MAX_CHECKPOINTS + 1, (num_curr,), device=self.device)
            self.num_checkpoints[curr_env_ids] = n_cps
            
            # 3. Select Generator Type
            if self.using_nursery_map:
                 # Override for Nursery: 3 CPs (Fixed), Simple Tracks and strictly no overlap
                 num_cps_nursery = self.config.num_checkpoints_fixed if self.config.num_checkpoints_fixed else 3
                 n_cps = torch.full((num_curr,), num_cps_nursery, device=self.device)
                 self.num_checkpoints[curr_env_ids] = n_cps
                 
                 # Generate for MAX_CHECKPOINTS but only use first n_cps
                 cps = TrackGenerator.generate_nursery_tracks(num_curr, MAX_CHECKPOINTS, WIDTH, HEIGHT, self.device)
                 # if num_reset > 0:
                 #     print(f"DEBUG: Generated NURSERY tracks for {num_curr} envs.")
                 self.checkpoints[curr_env_ids] = cps
            else:
                # Standard
                cps = TrackGenerator.generate_max_entropy(num_curr, MAX_CHECKPOINTS, WIDTH, HEIGHT, self.device)
                # if num_reset > 0:
                #      print(f"DEBUG: Generated MAX ENTROPY tracks for {num_curr} envs (CPs: {MIN_CHECKPOINTS}-{MAX_CHECKPOINTS}).")
                self.checkpoints[curr_env_ids] = cps
                # Standard
                cps = TrackGenerator.generate_max_entropy(num_curr, MAX_CHECKPOINTS, WIDTH, HEIGHT, self.device)
                # if num_reset > 0:
                #      print(f"DEBUG: Generated MAX ENTROPY tracks for {num_curr} envs (CPs: {MIN_CHECKPOINTS}-{MAX_CHECKPOINTS}).")
                self.checkpoints[curr_env_ids] = cps

            
        # 2. Reset Pods
        # Start at CP0
        start_pos = self.checkpoints[env_ids, 0] # [N, 2]
        
        # Look at CP1
        target_pos = self.checkpoints[env_ids, 1]
        angle_rad = torch.atan2(target_pos[:, 1] - start_pos[:, 1], target_pos[:, 0] - start_pos[:, 0])
        angle_deg = torch.rad2deg(angle_rad)
        
        # Position offsets to not overlap
        # 0: +disp, 1: -disp etc.
        # Perpendicular vector
        dx = target_pos[:, 0] - start_pos[:, 0]
        dy = target_pos[:, 1] - start_pos[:, 1]
        dist = torch.sqrt(dx**2 + dy**2) + 1e-5
        nx = -dy / dist
        ny = dx / dist
        
        # Pod 0: +400 * N
        # Pod 1: -400 * N
        # Pod 2: +1200 * N (Team 2 starts behind? Or side by side?)
        # Rules: "All pods start at the first checkpoint, facing the second."
        # Usually arranged in a grid or line.
        
        # Simple Line Arrangement
        offsets = torch.tensor(self.spawn_config.offsets, device=self.device)
        
        for i in range(4):
            # Reset Physics State
            self.physics.pos[env_ids, i, 0] = start_pos[:, 0] - nx * offsets[i] # Slightly behind/displaced
            self.physics.pos[env_ids, i, 1] = start_pos[:, 1] - ny * offsets[i]
            self.physics.vel[env_ids, i] = 0
            self.physics.angle[env_ids, i] = angle_deg
            self.physics.mass[env_ids, i] = 1.0
            self.physics.shield_cd[env_ids, i] = 0
            
            # --- Curriculum Logic (Spawn Control) ---
            # Stage 0 (Solo): Active=[0]. Others=[Infinity].
            # Stage 1 (Duel): Active=[0, 2]. Others=[Infinity].
            # Stage 2 (League): Active=[0, 1, 2, 3].
            
            active = True
            # --- Curriculum Logic (Spawn Control) ---
            active = False
            if i in self.config.active_pods:
                active = True
            
            if not active:
                # Move to infinity to avoid collision/observation noise
                self.physics.pos[env_ids, i, 0] = -100000.0
                self.physics.pos[env_ids, i, 1] = -100000.0
            
        self.physics.boost_available[env_ids] = True
        
        # Reset Game Logic
        self.next_cp_id[env_ids] = 1
        self.laps[env_ids] = 0
        
        self.timeouts[env_ids] = self.config.timeout_steps

             
        self.dones[env_ids] = False
        self.winners[env_ids] = -1
        self.cps_passed[env_ids] = 0
        self.steps_last_cp[env_ids] = 0
        
        # Update Prev Dist for rewards
        self.update_progress_metric(env_ids)
        
        # Reset Role Locks
        self.role_lock_timer[env_ids] = 0
        self._update_roles(env_ids)
        
        # Reset Ranks
        self.prev_ranks[env_ids] = self._get_ranks(env_ids)

    def set_stage(self, stage_id: int, config: EnvConfig, reset_env: bool = False):
        self.curriculum_stage = stage_id
        self.config = config
        self._update_map_mode()
        
        if reset_env:
            self.reset()

    def _update_map_mode(self):
        self.using_nursery_map = False
        if self.config.track_gen_type == "nursery":
            if self.config.mode_name == "nursery":
                self.using_nursery_map = True
            else:
                print(f"WARNING: Nursery Map requested but Mode is '{self.config.mode_name}'. Enforcing policy: FALLBACK TO MAX ENTROPY.")

    def _update_roles(self, env_ids):
        # Calculate Progress Score
        # Score = Laps * 1000 + NextCP * 10 + (1 - Dist/20000)
        # Higher is better
        
        # Gather data
        laps = self.laps[env_ids] # [N, 4]
        next_cp = self.next_cp_id[env_ids] # [N, 4]
        # Dist to next CP.
        # We can reuse self.prev_dist if valid, but reset might not have it yet?
        # reset calls update_progress_metric BEFORE this, so prev_dist is valid.
        dists = self.prev_dist[env_ids] # [N, 4]
        
        # Normalize dist to 0-1 (inverted, closer is better)
        # Max map dist approx 20000.
        dist_score = 1.0 - (dists / 20000.0)
        
        total_score = (laps * 1000.0) + (next_cp * 10.0) + dist_score
        
        # Compare Team 0: Pod 0 vs 1
        s0 = total_score[:, 0]
        s1 = total_score[:, 1]
        
        # Runner is argmax.
        # If s0 >= s1: Pod0 is Runner.
        # Tie-break: Pod0 (by index).
        runner0 = (s0 >= s1)
        
        # Team 1: Pod 2 vs 3
        s2 = total_score[:, 2]
        s3 = total_score[:, 3]
        runner2 = (s2 >= s3)
        
        # Hysteresis Logic
        # Filter for current envs
        current_timers = self.role_lock_timer[env_ids]
        can_update_mask = (current_timers == 0) # [len(env_ids)]
        
        # Get indices of envs that CAN update (subset of env_ids)
        # We need these to index global tensors like self.is_runner
        update_env_ids = env_ids[can_update_mask]
        
        if len(update_env_ids) > 0:
             # Extract new values for these specific envs
             new_r0 = runner0[can_update_mask]
             new_r2 = runner2[can_update_mask]
             
             # Extract old values for change detection
             old_r0 = self.is_runner[update_env_ids, 0]
             old_r2 = self.is_runner[update_env_ids, 2]
             
             # Update Global State
             self.is_runner[update_env_ids, 0] = new_r0
             self.is_runner[update_env_ids, 1] = ~new_r0
             self.is_runner[update_env_ids, 2] = new_r2
             self.is_runner[update_env_ids, 3] = ~new_r2
             
             # Change Detection
             changed0 = (old_r0 != new_r0)
             changed2 = (old_r2 != new_r2)
             
             env_changed = changed0 | changed2
             
             # Indices of envs that CHANGED
             changed_env_ids = update_env_ids[env_changed]
             
             if len(changed_env_ids) > 0:
                 self.role_lock_timer[changed_env_ids] = 50


    def update_progress_metric(self, env_ids):
        # Calculate distance to next checkpoint
        # gather next CP pos
        next_cp_indices = self.next_cp_id[env_ids] # [N, 4]
        # checkpoints: [N_total, MaxCP, 2] -> index with env_ids
        cps = self.checkpoints[env_ids] # [N, Max, 2]
        
        # Gather logic
        # We need [N, 4, 2] target positions
        # next_cp_indices is [N, 4].
        # Expand cps to gather?
        # B = len(env_ids)
        # gathered_cx = cps.gather(1, next_cp_indices)... dimension mismatch.
        # Manual lookup loop is easiest for 4 pods
        
        for i in range(4):
            cp_idx = next_cp_indices[:, i] # [N]
            # Select CP for each batch elt.
            # advanced indexing: cps[range, cp_idx]
            targets = cps[torch.arange(len(env_ids), device=self.device), cp_idx] # [N, 2]
            curr_pos = self.physics.pos[env_ids, i]
            dist = torch.norm(targets - curr_pos, dim=1)
            self.prev_dist[env_ids, i] = dist

    def step(self, actions, reward_weights, tau=0.0, team_spirit=0.0):
        """
        Stepping the Environment.
        actions: [B, 4, 4] (Thrust, Angle, Shift, Boost)
        reward_weights: [B, 13] (Per-environment weights, taken from Agent)
        tau: float, Dense Reward Annealing factor (0.0 = Full Dense, 1.0 = Full Sparse)
        team_spirit: float, 0.0=Selfish, 1.0=Cooperative. Blends rewards.
        """
        if reward_weights is None:
            # Construct default tensor
            reward_weights = torch.zeros((self.num_envs, 15), device=self.device)
            # Use default dict to fill 
            # Note: We can pre-compute this but for robust fallback:
            for k, v in DEFAULT_REWARD_WEIGHTS.items():
                reward_weights[:, k] = v
        
        # Unpack weights [B, 13]
        w_win = reward_weights[:, RW_WIN] # Sparse
        w_loss = reward_weights[:, RW_LOSS] # Sparse
        w_checkpoint = reward_weights[:, RW_CHECKPOINT] # Individual (Motor)
        w_chk_scale = reward_weights[:, RW_CHECKPOINT_SCALE] # Individual
        w_progress = reward_weights[:, RW_PROGRESS] # Individual (Replaces VELOCITY)
        w_col_run = reward_weights[:, RW_COLLISION_RUNNER] # Individual
        w_col_block = reward_weights[:, RW_COLLISION_BLOCKER] # Individual
        w_step_pen = reward_weights[:, RW_STEP_PENALTY] # Individual
        w_orient = reward_weights[:, RW_ORIENTATION] # Individual
        w_wrong_way = reward_weights[:, RW_WRONG_WAY] # Individual
        w_col_mate = reward_weights[:, RW_COLLISION_MATE] # Individual
        w_prox = reward_weights[:, RW_PROXIMITY] # Individual
        w_magnet = reward_weights[:, RW_MAGNET] # Individual (New)
        w_rank = reward_weights[:, RW_RANK] # Rank Change
        
        # --- Team Spirit Blending ---
        # Modify weights based on spirit
        # Spirit 0.0 (Selfish): Base Weights.
        # Spirit 1.0 (Cooperative): Boost Win/Loss, Reduce Checkpoint/Velocity (Individual stuff).
        
        # Win/Loss Multiplier: 1.0 + team_spirit (Doubles importance at full spirit)
        w_win = w_win * (1.0 + team_spirit)
        w_loss = w_loss * (1.0 + team_spirit)
        
        # Individual Multiplier: REMOVED penalty for Team Spirit.
        # We want robust navigation always, even if cooperative.
        # indiv_mult = 1.0 - (0.5 * team_spirit) 
        indiv_mult = 1.0 # Constant
        
        w_checkpoint = w_checkpoint * indiv_mult
        w_chk_scale = w_chk_scale * indiv_mult
        w_progress = w_progress * indiv_mult
        
        # Blocker/Team Interaction:
        # Blocker Reward IS a team contribution, so maybe boost it?
        # Collision Mate Penalty: Boost it (Don't hit friends!)
        w_col_mate = w_col_mate * (1.0 + team_spirit)
        
        # Unpack Actions & Clamp
        # Thrust: [0..1] -> [0..100]
        act_thrust = torch.clamp(actions[..., 0], 0.0, 1.0) * 100.0
        # Angle: [-1..1] -> [-18..18]
        act_angle = torch.clamp(actions[..., 1], -1.0, 1.0) * 18.0
        # Shield: > 0.5
        act_shield = actions[..., 2] > 0.5
        # Boost: > 0.5
        act_boost = actions[..., 3] > 0.5
        
        # --- Bot Logic (Same as before) ---
        if self.config.use_bots:
            bot_pods = self.config.bot_pods
            
            for bot_id in bot_pods:
                # Override Actions with Simple Bot
                # Bot: Steer towards next checkpoint
                
                # Get target
                opp_nid = self.next_cp_id[:, bot_id]
                batch_indices = torch.arange(self.num_envs, device=self.device)
                target = self.checkpoints[batch_indices, opp_nid]
                p_pos = self.physics.pos[:, bot_id]
                
                # Desired Angle
                diff = target - p_pos
                desired_rad = torch.atan2(diff[:, 1], diff[:, 0])
                desired_deg = torch.rad2deg(desired_rad)
                
                # --- Dynamic Difficulty Scaling ---
                # 1. Steering Noise
                noise_scale = (1.0 - self.bot_difficulty) * self.bot_config.difficulty_noise_scale
                noise = (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0) * noise_scale
                desired_deg += noise
                
                # Current Angle
                curr_deg = self.physics.angle[:, bot_id]
                
                # Delta
                delta = desired_deg - curr_deg
                # Normalize -180..180
                delta = (delta + 180) % 360 - 180
                
                # Clamp to -18..18
                delta = torch.clamp(delta, -18.0, 18.0)
                
                # Set Actions
                act_angle[:, bot_id] = delta
                
                # 2. Thrust Scaling
                # 20 + (80 * diff)
                thrust_val = self.bot_config.thrust_base + (self.bot_config.thrust_scale * self.bot_difficulty)
                act_thrust[:, bot_id] = thrust_val 
                
                act_shield[:, bot_id] = False
                act_boost[:, bot_id] = False
        
        # Track Previous Position for Continuous Collision Detection (CCD)
        prev_pos = self.physics.pos.clone()

        # 1. Physics Step
        collisions = self.physics.step(act_thrust, act_angle, act_shield, act_boost)
        
        # Calculate Collision Flags for Telemetry [Batch, 4]
        col_sums = collisions.sum(dim=2)
        collision_flags = (col_sums > 0).float()
        
        # 2. Reward Containers
        rewards_indiv = torch.zeros((self.num_envs, 4), device=self.device) # Pure Individual
        rewards_team = torch.zeros((self.num_envs, 4), device=self.device) # Pure Team (Win/Loss)

        # Metric Helpers
        infos = {
             "laps_completed": torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device),
             "checkpoints_passed": torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device),
             "current_streak": self.cps_passed.clone(),
             "cp_steps": torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device), # Steps taken for passed CPs
             "collision_flags": collision_flags # [Batch, 4]
        }

        # --- DENSE REWARDS (Individual) ---
        new_dists = torch.zeros_like(self.prev_dist)
        
        # Annealing
        dense_mult = (1.0 - tau)
        if isinstance(dense_mult, torch.Tensor):
             dense_mult = dense_mult.squeeze()
        
        for i in range(4):
            next_ids = self.next_cp_id[:, i]
            batch_indices = torch.arange(self.num_envs, device=self.device)
            target_pos = self.checkpoints[batch_indices, next_ids]
            curr_pos = self.physics.pos[:, i]
            
            # --- A. Progress Reward (Prev Dist - Curr Dist) ---
            new_dists[:, i] = torch.norm(target_pos - curr_pos, dim=1)
            
            # Positive = Approach, Negative = Retreat
            progress = self.prev_dist[:, i] - new_dists[:, i]
            
            # Apply Progress Reward
            # Note: 1 unit distance = 1 point if weight=1.0. 
            rewards_indiv[:, i] += progress * w_progress * dense_mult
            
            # --- B. Magnet Reward (Proximity Center) ---
            # If inside Approach Radius (e.g. 2 * Checkpoint Radius or just Checkpoint Radius?)
            # Let's say we pull them in from 1.5x Radius.
            MAGNET_RADIUS = CHECKPOINT_RADIUS * 1.5
            dist = new_dists[:, i]
            
            in_magnet_mask = dist < MAGNET_RADIUS
            if in_magnet_mask.any():
                # Score 0.0 to 1.0 (Close)
                magnet_score = (1.0 - (dist[in_magnet_mask] / MAGNET_RADIUS))
                # Square it to make the center much more attractive than the edge? No, linear is stable.
                rewards_indiv[in_magnet_mask, i] += magnet_score * w_magnet[in_magnet_mask] * dense_mult
        
        # --- C. Orientation Reward (Soft Guidance / Wrong Way) ---
        if w_orient.sum() > 0.0 or w_wrong_way.sum() > 0.0:
            for i in range(4):
                next_ids = self.next_cp_id[:, i]
                batch_indices = torch.arange(self.num_envs, device=self.device)
                target_pos = self.checkpoints[batch_indices, next_ids] 
                curr_pos = self.physics.pos[:, i] 
                
                diff = target_pos - curr_pos
                target_angle = torch.atan2(diff[:, 1], diff[:, 0])
                curr_angle = torch.deg2rad(self.physics.angle[:, i])
                alignment = torch.cos(target_angle - curr_angle)

                # Positive Reward (Soft Guidance)
                if w_orient.sum() > 0.0:
                    THRESHOLD = self.reward_scaling_config.orientation_threshold
                    pos_score = torch.clamp((alignment - THRESHOLD) / (1.0 - THRESHOLD), 0.0, 1.0)
                    rewards_indiv[:, i] += pos_score * w_orient * dense_mult

                # Negative Penalty (Wrong Way)
                neg_mask = alignment < -0.5 # Strictly facing away
                if neg_mask.any():
                    # Heavy penalty for wrong way
                    rewards_indiv[neg_mask, i] += alignment[neg_mask] * w_wrong_way[neg_mask]

        # --- D. Rank Reward (Potential-Based) ---
        # "Overtaking Reward"
        w_rank = reward_weights[:, RW_RANK]
        if w_rank.sum() > 0.0:
            # Calculate current ranks
            # Optimized: Pass slice(None) to avoid arange(num_envs) allocation
            curr_ranks = self._get_ranks(slice(None))
            
            # Potential diff: Prev - Curr
            # Rank 1 (2nd) -> Rank 0 (1st). Diff = 1 - 0 = +1 (Improvement)
            # Rank 0 (1st) -> Rank 1 (2nd). Diff = 0 - 1 = -1 (Loss)
            rank_diff = self.prev_ranks - curr_ranks
            
            # Apply Reward for each pod
            # Note: rank_diff is [N, 4]
            # w_rank is [N]. Expand.
            
            rewards_indiv += rank_diff.float() * w_rank.unsqueeze(1)
            
            # Update state
            self.prev_ranks = curr_ranks


        # --- D. Step Penalty (Constant) ---
        # Fixed negative reward per step to encourage speed.
        if w_step_pen.sum() > 0:
            active_mask = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
            for i in self.config.step_penalty_active_pods:
                active_mask[:, i] = True
                
            # Apply to active pods
            # w_step_pen is positive in config (e.g. 1.0), so we subtract it.
            # Assuming w_step_pen broadcastable [B]
            pen_val = w_step_pen.unsqueeze(1).expand(-1, 4)
            
            # Only apply where mask is true
            # rewards_indiv -= pen_val * active_mask.float()
            # Careful with shape
            penalty_tensor = pen_val * active_mask.float()
            rewards_indiv -= penalty_tensor

        
        # D. Checkpoints (Individual Progress)
        # For each pod, check distance to next_cp
        # Logic: dist < 600 -> passed.
        
        all_cp_pos = self.checkpoints # [B, Max, 2]
        
        events_cp_passed = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device) # For visuals/logging
        
        for i in range(4):
            next_ids = self.next_cp_id[:, i]
            # Gather targets
            # use unsqueeze exp.
            batch_indices = torch.arange(self.num_envs, device=self.device)
            target_pos = all_cp_pos[batch_indices, next_ids] # [B, 2]
            
            p_pos = self.physics.pos[:, i] # [B, 2]
            
            # --- Continuous Collision Detection (Segment-Circle Intersection) ---
            # Segment: prev_pos[:, i] -> p_pos
            # Circle: target_pos, Radius 600
            
            p_prev = prev_pos[:, i]
            
            # Vector A->B
            seg_v = p_pos - p_prev
            seg_len_sq = (seg_v ** 2).sum(dim=1)
            
            # Vector A->C (Center)
            ac_v = target_pos - p_prev
            
            # Project C onto Line AB: t = (AC . AB) / |AB|^2
            t = (ac_v * seg_v).sum(dim=1) / (seg_len_sq + 1e-6)
            
            # Clamp t to segment [0, 1]
            t = torch.clamp(t, 0.0, 1.0)
            
            # Closest Point
            closest = p_prev + seg_v * t.unsqueeze(1)
            
            # Dist Closest -> Center
            dist_sq = ((closest - target_pos) ** 2).sum(dim=1)
            
            # Check pass
            # Radius 600
            passed = dist_sq < (CHECKPOINT_RADIUS ** 2)
            
            # Update State
            if passed.any():
                pass_idx = torch.nonzero(passed).squeeze(-1)
                
                # Reset timeout
                self.timeouts[pass_idx, i] = self.config.timeout_steps 
                
                # Determine Next CP and Lap Logic
                # Use cached 'next_ids' (OLD target)
                old_target_id = next_ids[passed]
                
                # Standard increment
                new_target_id = (old_target_id + 1) 
                
                # Wrap
                limits = self.num_checkpoints[pass_idx] # [P]
                wrapped = new_target_id % limits
                
                # Apply Update
                self.next_cp_id[pass_idx, i] = wrapped
                
                # Check Lap Completion
                # If we passed CP 0 (Start/Finish Line), we completed a lap.
                # CP0 is ID 0.
                just_passed_zero = (old_target_id == 0)
                
                if just_passed_zero.any():
                    z_idx = pass_idx[just_passed_zero]
                    self.laps[z_idx, i] += 1
                    infos["laps_completed"][z_idx, i] = 1
                    
                    # --- Curriculum Metric: Solo Complete ---
                    if self.curriculum_stage == STAGE_NURSERY or self.curriculum_stage == STAGE_SOLO:
                        # If Pod 0 finished a lap
                        if i == 0:
                             self.stage_metrics["solo_completes"] += len(z_idx)
                

                
                # --- Metric: Checkpoints Passed ---
                # Only count for active pod (Pod 0 in Solo)?
                # Let's count all active pods to be fair metric of "system activity".
                # But user wants "progress".
                if self.curriculum_stage == STAGE_NURSERY or self.curriculum_stage == STAGE_SOLO:
                    if i == 0:
                         self.stage_metrics["checkpoint_hits"] += len(pass_idx)
                else:
                    self.stage_metrics["checkpoint_hits"] += len(pass_idx)

                # w.get("checkpoint", 500.0) -> Add to buffer
                # Progressive Reward: Base + (Streak * Scale)
                # Base is now 500.0 as requested.
                # Assuming increment is also 500.0?
                # Or use the weight as base?
                # original plan: base=4000. user: make base 500.
                # Let's say w["checkpoint"] is the BASE.
                # And we add a scaling factor.
                
                # w.get("checkpoint", 2000.0) -> Base
                # Progressive Reward: Base + (Streak * Scale) + DYNAMIC DECAY
                
                self.cps_passed[pass_idx, i] += 1
                infos["checkpoints_passed"][pass_idx, i] = 1
                streak = self.cps_passed[pass_idx, i].float() # 1, 2, 3...
                
                # Record Efficiency (Steps taken)
                taken_steps = self.steps_last_cp[pass_idx, i]
                infos["cp_steps"][pass_idx, i] = taken_steps
                self.steps_last_cp[pass_idx, i] = 0
                
                # --- REWARD CALCULATION ---
                # RW_LAP imported from config
                
                w_cp_base = w_checkpoint[pass_idx]
                w_lap_base = reward_weights[pass_idx, RW_LAP] if reward_weights.shape[1] > RW_LAP else torch.full((len(pass_idx),), 2000.0, device=self.device)
                
                # Split Logic: Lap vs Normal CP
                # 'just_passed_zero' is boolean mask for pass_idx
                
                # 1. Normal CP Rewards
                # Apply to ~just_passed_zero
                normal_mask = ~just_passed_zero
                
                # Calculate Normal Reward for ALL (we will mask later)
                streak_bonus = (streak - 1.0) * w_chk_scale[pass_idx]
                reward_normal = w_cp_base + streak_bonus
                
                # 2. Lap Rewards
                # Reward = Base * (Multiplier ^ LapIndex)
                # Note: self.laps was ALREADY incremented above for 'just_passed_zero' entries.
                # So we use current lap count (which is 1 for first lap, 2 for second...)
                # Wait, self.laps starts at 0. After crossing start line once (finish lap 1), it becomes 1.
                # So 'Lap 1' completion -> laps=1.
                # Formula: Base * (1.5 ** (laps - 1))?
                # Lap 1 (laps=1): Base * 1.5^0 = Base. Correct.
                # Lap 2 (laps=2): Base * 1.5^1 = Base * 1.5. Correct.
                current_laps = self.laps[pass_idx, i]
                lap_mult_pow = torch.clamp(current_laps - 1, min=0).float()
                
                # Multiplier
                mult = torch.pow(LAP_REWARD_MULTIPLIER, lap_mult_pow)
                reward_lap = w_lap_base * mult
                
                # Combine
                # If just_passed_zero: use reward_lap
                # Else: use reward_normal
                total_reward = torch.where(just_passed_zero, reward_lap, reward_normal)

                # --- TIME EXTENSION ---
                # Reset Timeout AFTER reward calc (so we used 'steps remaining' accurately)
                self.timeouts[pass_idx, i] = self.config.timeout_steps
                mate_idx = i ^ 1
                self.timeouts[pass_idx, mate_idx] = self.config.timeout_steps
                
                # DIRECT REWARD (Individual)
                # rewards[pass_idx, i] += total_reward
                rewards_indiv[pass_idx, i] += total_reward
                
                # Correction for Dense Reward Overshoot
                # If we passed, we reached the target (distance 0). 
                # The Dense Logic penalized us for moving "away" from it (the overshoot).
                # We add back the distance from target * scale to treat it as "Arrived at 0".
                # passed is bool mask [B]. pass_idx is indices.
                # new_dists is [B, 4].
                overshoot_dist = new_dists[pass_idx, i]
                
                # Scaling for Dense Reward (Explicit S_VEL)
                # Correction applies to POD i
                rewards_indiv[pass_idx, i] += overshoot_dist * w_progress[pass_idx] * dense_mult

        # Calculate dist_to_next for Nursery Metric [B, 4]
        # We need distance to next checkpoint for ALL pods, regardless of passing.
        # This is essentially 'new_dists' calculated earlier at line 447.
        infos['dist_to_next'] = new_dists
        
        # Increment Step Counter for Efficiency
        self.steps_last_cp += 1

        # 3. Timeouts (Loss Condition)
        self.timeouts -= 1
        timed_out = (self.timeouts <= 0)
        env_timed_out = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        if self.curriculum_stage == STAGE_SOLO:
            env_timed_out = timed_out[:, 0]
        elif self.curriculum_stage == STAGE_DUEL:
            # Only Agent (Pod 0) triggers timeout reset. 
            # Bot (Pod 2) timeout is ignored (DNF).
            env_timed_out = timed_out[:, 0]
        elif self.curriculum_stage == STAGE_TEAM:
            # Only Agents (Pod 0, 1) trigger timeout reset.
            # Bots (Pod 2, 3) timeout is ignored.
            env_timed_out = timed_out[:, 0] | timed_out[:, 1]
        else:
            # League or others: Any timeout resets?
            # Or should we be strict? 
            # In purely competitive (League), if one dies, maybe we reset?
            # Let's keep 'any' for League for now as use_bots=False.
            env_timed_out = timed_out.any(dim=1)

        self.dones = self.dones | env_timed_out
        
        # --- Degressive Timeout Penalty ---
        # Penalize agents for failing to finish, scaled by how much they failed.
        # Concept: "Billing for Wasted Time"
        
        if env_timed_out.any() and self.curriculum_stage != STAGE_NURSERY:
            # Identify indices
            idx = torch.nonzero(env_timed_out).squeeze(-1)
            
            # Apply to all 4 pods (if they timed out logic applies)
            # In Solo/Duel, 'env_timed_out' is OR of active pods.
            # We strictly penalize the specific pod that caused timeout? 
            # Or assume global timeout applies to all?
            # 'timed_out' is [B, 4]. Let's use that for precision.
            
            for p_i in range(4):
                pod_timeouts = timed_out[:, p_i]
                if pod_timeouts.any():
                     p_idx = torch.nonzero(pod_timeouts).squeeze(-1)
                     
                     # Calculate Progress
                     passed = self.cps_passed[p_idx, p_i].float()
                     
                     # Total Goal (Checkpoints * Laps)
                     # self.num_checkpoints is [B]
                     total_goal = (self.num_checkpoints[p_idx] * MAX_LAPS).float()
                     
                     progress = torch.clamp(passed / total_goal, 0.0, 1.0)
                     
                     # Fixed Rate 25.0 * 100 Steps = 2500.0
                     # [CRITICAL UPDATE] Timeout Penalty must exceed Loss Penalty (5000.0)
                     # otherwise agents prefer to spin/stall than race and lose.
                     if self.curriculum_stage == STAGE_DUEL:
                         MAX_PENALTY = TIMEOUT_PENALTY_DUEL # Huge penalty to force finishing
                     else:
                         MAX_PENALTY = TIMEOUT_PENALTY_STANDARD
                     
                     penalty = MAX_PENALTY * (1.0 - progress)
                     
                     rewards_indiv[p_idx, p_i] -= penalty
        finished = (self.laps >= MAX_LAPS)
        
        if finished.any():
            env_won = finished.any(dim=1)
            winner_indices = torch.nonzero(env_won).squeeze(-1)
            
            t0_wins = (finished[:, 0] | finished[:, 1])
            t1_wins = (finished[:, 2] | finished[:, 3])
            
            self.winners[t0_wins] = 0
            self.winners[t1_wins & ~t0_wins] = 1
            
            self.dones[env_won] = True
            
            # --- SPARSE REWARDS (Team Shared) ---
            # 1. Win/Loss is inherently Team-based.
            # 2. We apply to 'rewards_team'.
            
            mask_w0 = (self.winners == 0) & env_won
            mask_w1 = (self.winners == 1) & env_won
            
            # Team 0 Rewards
            rewards_team[mask_w0, 0] += w_win[mask_w0]
            rewards_team[mask_w0, 1] += w_win[mask_w0]
            rewards_team[mask_w0, 2] -= w_loss[mask_w0]
            rewards_team[mask_w0, 3] -= w_loss[mask_w0]
            
            # Team 1 Rewards
            rewards_team[mask_w1, 2] += w_win[mask_w1]
            rewards_team[mask_w1, 3] += w_win[mask_w1]
            rewards_team[mask_w1, 0] -= w_loss[mask_w1]
            rewards_team[mask_w1, 1] -= w_loss[mask_w1]
            
            # Metrics
            if self.curriculum_stage == STAGE_DUEL:
                n_wins = mask_w0.sum().item()
                n_games = env_won.sum().item()
                self.stage_metrics["duel_wins"] += n_wins
                self.stage_metrics["duel_games"] += n_games
                self.stage_metrics["recent_wins"] += n_wins
                self.stage_metrics["recent_games"] += n_games
            elif self.curriculum_stage == STAGE_TEAM:
                n_wins = mask_w0.sum().item() 
                n_games = env_won.sum().item() 
                self.stage_metrics["recent_wins"] += n_wins
                self.stage_metrics["recent_games"] += n_games
        
        S_VEL = 1.0 / 1000.0
        self.update_progress_metric(torch.arange(self.num_envs, device=self.device))
        
        num_dones = self.dones.sum().item()
        if num_dones > 0:
             self.stage_metrics["recent_episodes"] += num_dones
        
        # --- Role Collision Rewards (Individual) ---
        runner_velocity_metric = torch.zeros((self.num_envs, 4), device=self.device)
        blocker_damage_metric = torch.zeros((self.num_envs, 4), device=self.device)
        
        for i in range(4):
            team = i // 2
            enemy_team = 1 - team
            enemy_indices = [2*enemy_team, 2*enemy_team + 1]
            
            is_run = self.is_runner[:, i] # [B]
            
            v_mag = torch.norm(self.physics.vel[:, i], dim=1)
            runner_velocity_metric[:, i] = v_mag * is_run.float() * S_VEL 
            
            impact_e1 = collisions[:, i, enemy_indices[0]]
            impact_e2 = collisions[:, i, enemy_indices[1]]
            total_impact = impact_e1 + impact_e2
            
            # 1. Runner Penalty
            runner_pen = -w_col_run * total_impact
            rewards_indiv[:, i] += runner_pen * is_run.float()
            
            # 2. Blocker Bonus
            is_block = ~is_run
            enemy_runner_mask = self.is_runner[:, enemy_indices] # [B, 2]
            
            bonus = torch.zeros(self.num_envs, device=self.device)
            bonus += impact_e1 * enemy_runner_mask[:, 0].float()
            bonus += impact_e2 * enemy_runner_mask[:, 1].float()
            
            blocker_damage_metric[:, i] = bonus * is_block.float()
            
            blocker_reward = w_col_block * bonus
            rewards_indiv[:, i] += blocker_reward * is_block.float()
        
            # 3. Teammate Collision Penalty
            mate_idx = i ^ 1
            impact_mate = collisions[:, i, mate_idx]
            mate_pen = -w_col_mate * impact_mate
            rewards_indiv[:, i] += mate_pen
            
        # --- Proximity Reward (Blocker -> Enemy Runner) ---
        if self.curriculum_stage >= STAGE_DUEL:
             for i in range(4):
                 is_block = ~self.is_runner[:, i]
                 if not is_block.any(): continue
                 
                 team = i // 2
                 enemy_team = 1 - team
                 enemy_indices = torch.tensor([2*enemy_team, 2*enemy_team + 1], device=self.device)
                 
                 e_runner_mask = self.is_runner[:, enemy_indices]
                 prox_rew = torch.zeros(self.num_envs, device=self.device)
                 p_pos = self.physics.pos[:, i]
                 
                 for e_idx in [0, 1]:
                      real_e_idx = enemy_indices[e_idx] # tensor
                      real_e = 2 * enemy_team + e_idx
                      e_pos = self.physics.pos[:, real_e]
                      dist = torch.norm(e_pos - p_pos, dim=1)
                      PROX_RADIUS = 3000.0
                      bonus = torch.clamp(1.0 - (dist / PROX_RADIUS), 0.0, 1.0)
                      # Only if active blocker and target is active runner
                      
                      # Fix mask logic:
                      # We need 'e_runner_mask' to match batch?
                      # e_runner_mask is [B, 2].
                      is_e_run = self.is_runner[:, real_e] # [B]
                      
                      condition = is_block & is_e_run
                      prox_rew[condition] += bonus[condition]
                 
                 rewards_indiv[:, i] += prox_rew * w_prox * is_block.float()
                 
        # Decrement Role Timer
        self.role_lock_timer = torch.clamp(self.role_lock_timer - 1, min=0)
            
        # Update Roles for Next Step
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._update_roles(all_ids)
        
        # Add Metrics to Info
        infos['runner_velocity'] = runner_velocity_metric
        infos['blocker_damage'] = blocker_damage_metric

        # --- FINAL BLENDING ---
        # Hybrid Reward:
        # R_total = R_indiv + Blend(R_team, spirit)
        final_rewards = rewards_indiv + rewards_team
        
        return final_rewards, self.dones.clone(), infos

    def get_obs(self):
        """
        Produce Observation Tensor for DeepSets (Body-Frame / Ego-Centric).
        Vectorized across B environments AND 4 Pods.
        
        Returns:
            self_obs: [B, 4, 14]
            entity_obs: [B, 4, 3, 13]
            cp_obs: [B, 4, 6]
        """
        
        B = self.num_envs
        device = self.device
        
        # Norm Constants
        S_POS = 1.0 / 16000.0
        S_VEL = 1.0 / 1000.0
        
        # 1. Physics State [B, 4]
        pos = self.physics.pos # [B, 4, 2]
        vel = self.physics.vel # [B, 4, 2]
        angle = self.physics.angle # [B, 4]
        
        # Rotation Helper (Vectorized for [..., 2] and [..., 1] angle)
        def rotate_vec(v, a_deg):
            # v: [..., 2], a_deg: [...]
            rad = torch.deg2rad(a_deg).unsqueeze(-1) # [..., 1]
            c = torch.cos(rad)
            s = torch.sin(rad)
            # Rotation Matrix [[c, s], [-s, c]]
            # x' = x*c + y*s
            # y' = -x*s + y*c
            vx = v[..., 0:1]
            vy = v[..., 1:2]
            nx = vx * c + vy * s
            ny = -vx * s + vy * c
            return torch.cat([nx, ny], dim=-1) # [..., 2]

        # --- Self Features (14) ---
        # 1. Local Velocity
        # Rel to own angle
        v_local = rotate_vec(vel, angle) * S_VEL # [B, 4, 2]
        
        # 2. Target Vector
        # Gather Target Pos
        # self.next_cp_id is [B, 4]
        # self.checkpoints is [B, N_CP, 2]
        # We need efficient gather.
        # Flatten batch:
        # flat_cp_ids = self.next_cp_id.long() + torch.arange(B, device=device).unsqueeze(1) * self.num_checkpoints.unsqueeze(1) # This assumes fixed N_CP? 
        # Wait, self.checkpoints is [B, 5, 2] max?
        # Actually checkpoints tensor is padded fixed size?
        # Let's check init... self.checkpoints = torch.zeros((num_envs, max_checkpoints, 2))
        # So we can index normally.
        
        # Correct Gather:
        # We want [B, 4, 2]
        # batch_idx [B, 1] broadcast to [B, 4]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, 4)
        target_pos = self.checkpoints[batch_idx, self.next_cp_id.long()] # [B, 4, 2]
        
        t_vec_g = target_pos - pos
        dest = torch.norm(t_vec_g, dim=-1, keepdim=True) * S_POS # [B, 4, 1]
        t_vec_l = rotate_vec(t_vec_g, angle) * S_POS # [B, 4, 2]
        
        # Alignment
        # t_vec_l is (Fwd, Right) scaled.
        # Cos = Fwd / Dist, Sin = Right / Dist
        dist_safe = dest + 1e-6
        align = t_vec_l / dist_safe # [B, 4, 2] (Cos, Sin)
        
        # Scalars
        shield = (self.physics.shield_cd.float() / 3.0).unsqueeze(-1) # [B, 4, 1]
        
        # Boost (Team Shared)
        # team 0: pods 0,1. team 1: pods 2,3.
        # boost_available is [B, 2]
        # Expand to [B, 4]
        team_idx = torch.tensor([0, 0, 1, 1], device=device).expand(B, 4)
        boost = self.physics.boost_available.gather(1, team_idx).float().unsqueeze(-1) # [B, 4, 1]
        
        timeout = (self.timeouts.float() / 100.0).unsqueeze(-1)
        lap = (self.laps.float() / 3.0).unsqueeze(-1)
        leader = self.is_runner.float().unsqueeze(-1)
        v_mag = torch.norm(vel, dim=-1, keepdim=True) * S_VEL
        
        # --- Rank Calculation ---
        # Score = Lap * 50000 + NextCP * 500 + (20000 - Dist)
        # Dist is 'dest' (Normalized distance / S_POS = True Distance)
        # dest is [B, 4, 1]. S_POS = 1/16000. So True Dist = dest * 16000.
        true_dist = dest.squeeze(-1) / S_POS
        
        # Laps and NextCP
        p_laps = self.laps.float()
        p_ncp = self.next_cp_id.float()
        
        # Score [B, 4]
        scores = p_laps * 50000.0 + p_ncp * 500.0 + (20000.0 - true_dist)
        
        # Calculate Rank (How many scores > my score)
        # Compare [B, 4, 1] vs [B, 1, 4] -> [B, 4, 4]
        s_col = scores.unsqueeze(2)
        s_row = scores.unsqueeze(1)
        # Count > Self (dim 2 sum)
        ranks = (s_row > s_col).sum(dim=2).float().unsqueeze(-1) # [B, 4, 1]
        
        rank_norm = ranks / 3.0 # Normalize 0 (1st) to 1 (4th) range [0, 1]
        pad = torch.zeros_like(v_mag)

        # Assemble Self
        # [B, 4, 15] (+1 Rank, +1 Pad retained)
        # v_local: 2, t_vec_l: 2, dest: 1, align: 2, shield, boost, timeout, lap, leader, v_mag, pad, rank
        self_obs = torch.cat([
            v_local, t_vec_l, dest, align, shield, boost, timeout, lap, leader, v_mag, pad, rank_norm
        ], dim=-1)
        
        # --- Entity Features (3 x 13) ---
        # We need "Others" for each Pod.
        # Indices map:
        # 0 -> 1, 2, 3
        # 1 -> 0, 2, 3
        # 2 -> 0, 1, 3
        # 3 -> 0, 1, 2
        other_indices = torch.tensor([
            [1, 2, 3], # 0: Mate, E, E
            [0, 2, 3], # 1: Mate, E, E
            [3, 0, 1], # 2: Mate, E, E
            [2, 0, 1]  # 3: Mate, E, E
        ], device=device) # [4, 3]
        
        # Broadcast to [B, 4, 3]
        # We can gather from pos [B, 4, 2]
        # Map: For each pod i, get 3 others.
        
        # Gather logic:
        # We want [B, 4, 3, 2] (Pos of others)
        # Expand Pos to [B, 1, 4, 2] -> [B, 4, 4, 2] ?
        # Or just fancy indexing.
        # pos[:, other_indices] works if pos is [B, 4, 2] -> result [B, 4, 3, 2]
        # (PyTorch simple indexing works on specific dim if others are :)
        # Actually `pos[:, other_indices]` might interpret as batch index?
        # Careful.
        
        # Let's perform Gather on dim 1.
        # We need index tensor of shape [B, 4, 3]
        # gather_idx = other_indices.unsqueeze(0).expand(B, -1, -1) # [B, 4, 3]
        # Gather requires index to have same dims as input except on gather dim.
        # Input [B, 4, 2]. We need to gather (dim 1). 
        # Gather index must be [B, 4, 3] -> Output [B, 4, 3, 2]? 
        # gather on dim=1, index must broadcast to [B, 4, 3, 2]? No.
        # torch.gather(input, dim, index)
        # index shape determines output shape.
        # We want to gather 2D coords.
        # Gather index must capture the last dim? No gather works on one dim.
        
        # Easier: Flatten B*4
        # But B is large.
        
        # Alternative: pos is [B, 4, 2].
        # We want o_pos [B, 4, 3, 2].
        # Use simple indexing:
        # o_pos = pos[:, other_indices] # This works mostly in numpy. In torch?
        # Let's try explicit expansion.
        o_pos = pos[:, other_indices] # [B, 4, 3, 2] assuming standard advanced indexing behavior
        o_vel = vel[:, other_indices]
        o_angle = angle[:, other_indices] # [B, 4, 3]
        o_shield = self.physics.shield_cd[:, other_indices] > 0 # [B, 4, 3]
        
        # Current Pod State, Expanded
        # p_pos: [B, 4, 1, 2]
        p_pos = pos.unsqueeze(2) 
        p_vel = vel.unsqueeze(2)
        p_angle = angle.unsqueeze(2) # [B, 4, 1]
        
        # Deltas
        d_pos_g = o_pos - p_pos # [B, 4, 3, 2]
        d_vel_g = o_vel - p_vel
        
        # Rotate all at once
        # p_angle is [B, 4, 1]. Need to match d_pos_g [B, 4, 3, 2].
        # Expand angle to [B, 4, 3]
        p_angle_exp = p_angle.expand(-1, -1, 3) 
        
        dp_local = rotate_vec(d_pos_g, p_angle_exp) * S_POS # [B, 4, 3, 2] (Fwd, Right)
        dv_local = rotate_vec(d_vel_g, p_angle_exp) * S_VEL
        
        # Relative Angle
        # o_angle: [B, 4, 3], p_angle: [B, 4, 1]
        # Broadcasts to [B, 4, 3]
        rel_angle = o_angle - p_angle # [B, 4, 3]
        rel_rad = torch.deg2rad(rel_angle).unsqueeze(-1)
        rel_cos = torch.cos(rel_rad) # [B, 4, 3, 1]
        rel_sin = torch.sin(rel_rad)
        
        # Dist
        dist = torch.norm(d_pos_g, dim=-1, keepdim=True) * S_POS # [B, 4, 3, 1]
        
        # Mate
        # o_idx: [B, 4, 3] (0..3)
        # team = idx // 2
        # Use other_indices [4, 3]
        o_team = other_indices // 2
        p_team = torch.arange(4, device=device).unsqueeze(1) // 2 # [4, 1]
        is_mate = (o_team == p_team).float().unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1) # [B, 4, 3, 1]
        
        # Shield
        o_shield_f = o_shield.float().unsqueeze(-1) # [B, 4, 3, 1]
        
        # Their Target (Relative to Me)
        # o_nid = next_cp_id gathered
        # next_cp_id [B, 4]
        # Gather for others
        o_nid = self.next_cp_id[:, other_indices] # [B, 4, 3]
        o_target = self.checkpoints[batch_idx.unsqueeze(2).expand(-1,-1,3), o_nid] # [B, 4, 3, 2]
        
        # p_pos is [B, 4, 1, 2]. Broadcasts fine against [B, 4, 3, 2]
        ot_vec_g = o_target - p_pos 
        
        # Rotate using expanded p_angle
        ot_vec_l = rotate_vec(ot_vec_g, p_angle_exp) * S_POS
        
        # Padding [B, 4, 3, 2]
        pad2 = torch.zeros_like(ot_vec_l)
        
        # Concat Entity
        # dp(2), dv(2), cos(1), sin(1), dist(1), mate(1), shield(1), ot(2), pad(2) -> 13
        entity_obs = torch.cat([
            dp_local, dv_local, rel_cos, rel_sin, dist, is_mate, o_shield_f, ot_vec_l, pad2
        ], dim=-1) # [B, 4, 3, 13]
        
        # --- CP Features (6) ---
        # Next (CP1) -> Next+1 (CP2)
        # Vector CP1->CP2 in My Body Frame
        # CP1 is target_pos [B, 4, 2]
        # t_vec_l is CP1 relative to Me [B, 4, 2]
        
        # Get CP2
        cp1_ids = self.next_cp_id.long()
        cp2_ids = (cp1_ids + 1) % self.num_checkpoints.unsqueeze(1)
        cp2_pos = self.checkpoints[batch_idx, cp2_ids] # [B, 4, 2]
        
        v12_g = cp2_pos - target_pos # [B, 4, 2]
        v12_l = rotate_vec(v12_g, angle) * S_POS
        
        pad_cp = torch.zeros((B, 4, 2), device=device)
        
        # t_vec_l (CP relative to Me), v12_l (CP1->CP2 relative to Me), Pad
        cp_obs = torch.cat([
            t_vec_l, v12_l, pad_cp
        ], dim=-1) # [B, 4, 6]
        
        # Permute to [4, B, ...] for contiguous memory access per Pod
        # self_obs: [B, 4, 14] -> [4, B, 14]
        self_c = self_obs.permute(1, 0, 2).contiguous()
        
        # entity_obs: [B, 4, 3, 13]
        # Split into Teammate (Index 0) and Enemies (Indices 1,2)
        # Teammate: [B, 4, 1, 13] -> Squeeze -> [B, 4, 13]
        tm_obs = entity_obs[:, :, 0, :]
        en_obs = entity_obs[:, :, 1:, :] # [B, 4, 2, 13]
        
        # Permute
        tm_c = tm_obs.permute(1, 0, 2).contiguous()
        en_c = en_obs.permute(1, 0, 2, 3).contiguous()
        
        # cp_obs: [B, 4, 6] -> [4, B, 6]
        cp_c = cp_obs.permute(1, 0, 2).contiguous()
        
        return self_c, tm_c, en_c, cp_c
