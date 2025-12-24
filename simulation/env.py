import torch
import math
from simulation.gpu_physics import GPUPhysics
from config import *

# Reward Indices
RW_WIN = 0
RW_LOSS = 1
RW_CHECKPOINT = 2
RW_CHECKPOINT_SCALE = 3
RW_VELOCITY = 4
RW_COLLISION_RUNNER = 5
RW_COLLISION_BLOCKER = 6
RW_STEP_PENALTY = 7
RW_ORIENTATION = 8
RW_WRONG_WAY = 9

DEFAULT_REWARD_WEIGHTS = {
    RW_WIN: 10000.0,
    RW_LOSS: 5000.0,
    RW_CHECKPOINT: 2000.0,
    RW_CHECKPOINT_SCALE: 50.0, # Kept as is, minor influence
    RW_VELOCITY: 0.05, # Drastically reduced to prevent "money printing"
    RW_COLLISION_RUNNER: 0.5,
    RW_COLLISION_BLOCKER: 2.0,
    RW_STEP_PENALTY: 10.0, # Reduced to balance with lower velocity
    RW_ORIENTATION: 0.005, # SOTA: De-emphasized to prevent "Orientation Trap". Implicit in velocity.
    RW_WRONG_WAY: 10.0
}

class PodRacerEnv:
    def __init__(self, num_envs, device='cuda', start_stage=STAGE_SOLO):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.physics = GPUPhysics(num_envs, device=device)
        self.curriculum_stage = start_stage # 0=Solo, 1=Duel, 2=League
        
        # Game State
        self.next_cp_id = torch.ones((num_envs, 4), dtype=torch.long, device=self.device) # Start at 1 (0 is start)
        self.laps = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device)
        self.timeouts = torch.full((num_envs, 4), TIMEOUT_STEPS, dtype=torch.long, device=self.device)
        
        # Checkpoints [Batch, MaxCP, 2]
        self.checkpoints = torch.zeros((num_envs, MAX_CHECKPOINTS, 2), device=self.device)
        self.num_checkpoints = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        
        # Rewards / Done
        self.rewards = torch.zeros((num_envs, 2), device=self.device) # Team 0, Team 1
        self.cp_reward_buffer = torch.zeros((num_envs, 4), device=self.device) # Buffer for spreading CP rewards
        self.cps_passed = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device) # Track CPs passed for scaling
        self.dones = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self.winners = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        
        # Roles
        self.is_runner = torch.zeros((num_envs, 4), dtype=torch.bool, device=self.device)

        
        # Progress tracking for dense rewards
        self.prev_dist = torch.zeros((num_envs, 4), device=self.device)
        self.steps_last_cp = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device) # Track steps per CP
        self.stage_metrics = {
            "solo_completes": 0,
            "solo_steps": 0,
            "checkpoint_hits": 0,
            "duel_wins": 0,
            "duel_games": 0,
            "recent_wins": 0,
            "recent_games": 0
        }
        
        self.bot_difficulty = 0.0 # 0.0 to 1.0


        
        self.reset()
        
    def reset(self, env_ids=None):
        if env_ids is None:
            # All
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # 1. Generate Tracks (Vectorized Retry Logic)
        # Optimized: Fail fast (3 attempts, 20 loops) -> Fallback to Procedural Path
        
        MAX_MAP_ATTEMPTS = 3
        BORDER_BUFFER = 2500.0
        MIN_DIST = 3000.0
        
        active_mask = torch.ones(num_reset, dtype=torch.bool, device=self.device)
        
        for attempt in range(MAX_MAP_ATTEMPTS):
            if not active_mask.any():
                break
                
            # Indices of envs to generate this round (relative to env_ids)
            pending_idx = torch.nonzero(active_mask).squeeze(-1)
            # Global indices to update
            curr_env_ids = env_ids[pending_idx]
            num_curr = len(curr_env_ids)
            
            # --- Generation Logic for curr_env_ids ---
            
            # 1. Reset CPs for these envs to 0
            self.checkpoints[curr_env_ids] = 0.0
            
            # 2. Random N CPs
            n_cps = torch.randint(MIN_CHECKPOINTS, MAX_CHECKPOINTS + 1, (num_curr,), device=self.device)
            self.num_checkpoints[curr_env_ids] = n_cps
            
            # 3. CP 0 (Start)
            cx = torch.rand(num_curr, device=self.device) * (WIDTH - 2*BORDER_BUFFER) + BORDER_BUFFER
            cy = torch.rand(num_curr, device=self.device) * (HEIGHT - 2*BORDER_BUFFER) + BORDER_BUFFER
            self.checkpoints[curr_env_ids, 0, 0] = cx
            self.checkpoints[curr_env_ids, 0, 1] = cy
            
            # 4. Generate Remaining CPs
            map_failed = torch.zeros(num_curr, dtype=torch.bool, device=self.device)
            
            for i in range(1, MAX_CHECKPOINTS):
                vals_needed = i < n_cps
                if not vals_needed.any():
                    break
                    
                need_mask = vals_needed & (~map_failed)
                if not need_mask.any():
                     continue

                valid_placement = torch.zeros(num_curr, dtype=torch.bool, device=self.device)
                BATCH_SIZE = 32
                MAX_PLACEMENT_LOOPS = 5 # Reduced for Speed
                
                for loop in range(MAX_PLACEMENT_LOOPS):
                    candidates_needed = need_mask & (~valid_placement)
                    
                    if not candidates_needed.any():
                        break
                        
                    # Generate Candidates
                    if self.curriculum_stage == STAGE_SOLO:
                         prev_cp = self.checkpoints[curr_env_ids, i-1].unsqueeze(1) 
                         theta = torch.rand(num_curr, BATCH_SIZE, device=self.device) * 2 * math.pi
                         r = torch.rand(num_curr, BATCH_SIZE, device=self.device) * 3000.0 + 3000.0
                         off_x = r * torch.cos(theta)
                         off_y = r * torch.sin(theta)
                         cands = prev_cp + torch.stack([off_x, off_y], dim=2)
                         
                         in_bounds = (cands[..., 0] > BORDER_BUFFER) & (cands[..., 0] < WIDTH - BORDER_BUFFER) & \
                                     (cands[..., 1] > BORDER_BUFFER) & (cands[..., 1] < HEIGHT - BORDER_BUFFER)
                    else:
                         rx = torch.rand(num_curr, BATCH_SIZE, device=self.device) * (WIDTH - 2*BORDER_BUFFER) + BORDER_BUFFER
                         ry = torch.rand(num_curr, BATCH_SIZE, device=self.device) * (HEIGHT - 2*BORDER_BUFFER) + BORDER_BUFFER
                         cands = torch.stack([rx, ry], dim=2)
                         in_bounds = torch.ones(num_curr, BATCH_SIZE, dtype=torch.bool, device=self.device)
                         
                    is_safe = in_bounds.clone()
                    for j in range(i):
                        prev = self.checkpoints[curr_env_ids, j].unsqueeze(1)
                        diff = cands - prev
                        dist_sq = (diff ** 2).sum(dim=2)
                        is_safe = is_safe & (dist_sq > MIN_DIST**2)
                        
                    has_valid = is_safe.any(dim=1)
                    success_mask = candidates_needed & has_valid
                    
                    if success_mask.any():
                        valid_indices = is_safe.long().argmax(dim=1)
                        sel_idx = valid_indices.view(-1, 1, 1).expand(-1, 1, 2)
                        winners = cands.gather(1, sel_idx).squeeze(1)
                        succ_idx_global = torch.nonzero(success_mask).squeeze(-1)
                        self.checkpoints[curr_env_ids[succ_idx_global], i] = winners[succ_idx_global]
                        valid_placement = valid_placement | success_mask
                
                failed_this_step = need_mask & (~valid_placement)
                if failed_this_step.any():
                    map_failed = map_failed | failed_this_step
            
            # Integrity Check
            for k_local in range(num_curr):
               if not map_failed[k_local]:
                   cnt = n_cps[k_local]
                   cps = self.checkpoints[curr_env_ids[k_local], :cnt]
                   if (cps == 0).all(dim=1).any():
                       map_failed[k_local] = True
            
            active_mask[pending_idx] = map_failed

        # Fallback: Procedural Path (Guaranteed Speed & Validity)
        # Instead of a straight line, we generate a zig-zag or structured path.
        if active_mask.any():
            final_fail_idx = torch.nonzero(active_mask).squeeze(-1)
            fail_env_ids = env_ids[final_fail_idx]
            n_fail = len(fail_env_ids)
            
            # Set to 5 checkpoints for simplicity in fallback
            self.num_checkpoints[fail_env_ids] = 5
            self.checkpoints[fail_env_ids] = 0.0
            
            # Procedural Generation: Zig-Zag
            # Start Left-Center
            start_x = 2500.0
            start_y = HEIGHT / 2.0
            
            # X Step: 3000
            # Y Amplitude: +/- 2000
            
            # Generate random Y-direction per env: -1 or 1
            y_dir = (torch.randint(0, 2, (n_fail,), device=self.device) * 2 - 1).float()
            
            for i in range(5):
                px = start_x + (i * 3000.0)
                # Alternate up/down offset
                offset = (i % 2) * 2500.0 * y_dir # 0, 2500, 0, 2500...
                
                # Jitter (small random noise) to prevent overfitting
                jitter_x = (torch.rand(n_fail, device=self.device) * 500.0) - 250.0
                jitter_y = (torch.rand(n_fail, device=self.device) * 500.0) - 250.0
                
                py = start_y + offset + jitter_y
                px = px + jitter_x
                
                # Clamp to be safe
                px = torch.clamp(px, BORDER_BUFFER, WIDTH - BORDER_BUFFER)
                py = torch.clamp(py, BORDER_BUFFER, HEIGHT - BORDER_BUFFER)
                
                self.checkpoints[fail_env_ids, i, 0] = px
                self.checkpoints[fail_env_ids, i, 1] = py
            
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
        offsets = torch.tensor([500, -500, 1500, -1500], device=self.device)
        
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
            if self.curriculum_stage == STAGE_SOLO:
                if i != 0: active = False
            elif self.curriculum_stage == STAGE_DUEL:
                if i != 0 and i != 2: active = False
            
            if not active:
                # Move to infinity to avoid collision/observation noise
                self.physics.pos[env_ids, i, 0] = -100000.0
                self.physics.pos[env_ids, i, 1] = -100000.0
            
        self.physics.boost_available[env_ids] = True
        
        # Reset Game Logic
        self.next_cp_id[env_ids] = 1
        self.laps[env_ids] = 0
        self.timeouts[env_ids] = TIMEOUT_STEPS
        self.dones[env_ids] = False
        self.winners[env_ids] = -1
        self.cp_reward_buffer[env_ids] = 0.0
        self.cps_passed[env_ids] = 0
        self.steps_last_cp[env_ids] = 0
        
        # Update Prev Dist for rewards
        self.update_progress_metric(env_ids)
        self._update_roles(env_ids)

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
        
        # Set Flags
        # Pod 0
        self.is_runner[env_ids, 0] = runner0
        # Pod 1
        self.is_runner[env_ids, 1] = ~runner0
        # Pod 2
        self.is_runner[env_ids, 2] = runner2
        # Pod 3
        self.is_runner[env_ids, 3] = ~runner2


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

    def step(self, actions, reward_weights=None, tau=0.0, beta=0.0):
        """
        actions: [Batch, 4, 4] -> [Thrust, Angle, Shield, Boost]
        reward_weights: Tensor [Batch, 10] (Optional, defaults to DEFAULT_REWARD_WEIGHTS)
        tau: Scalar (Dense -> Sparse annealing)
        beta: Scalar (Selfish -> Team annealing)
        """
        if reward_weights is None:
            # Construct default tensor
            reward_weights = torch.zeros((self.num_envs, 10), device=self.device)
            # Use default dict to fill 
            # Note: We can pre-compute this but for robust fallback:
            for k, v in DEFAULT_REWARD_WEIGHTS.items():
                reward_weights[:, k] = v
        
        # Aliases for readability
        w_win = reward_weights[:, RW_WIN]
        w_loss = reward_weights[:, RW_LOSS]
        w_checkpoint = reward_weights[:, RW_CHECKPOINT]
        w_chk_scale = reward_weights[:, RW_CHECKPOINT_SCALE]
        w_velocity = reward_weights[:, RW_VELOCITY]
        w_col_run = reward_weights[:, RW_COLLISION_RUNNER]
        w_col_block = reward_weights[:, RW_COLLISION_BLOCKER]
        w_step_pen = reward_weights[:, RW_STEP_PENALTY]
        w_orient = reward_weights[:, RW_ORIENTATION]
        w_wrong_way = reward_weights[:, RW_WRONG_WAY]

        # Unpack Actions & Clamp
        # Thrust: [0..1] -> [0..100]
        act_thrust = torch.clamp(actions[..., 0], 0.0, 1.0) * 100.0
        # Angle: [-1..1] -> [-18..18]
        act_angle = torch.clamp(actions[..., 1], -1.0, 1.0) * 18.0
        # Shield: > 0.5
        act_shield = actions[..., 2] > 0.5
        # Boost: > 0.5
        act_boost = actions[..., 3] > 0.5
        
        # --- Stage 2 (Duel) Bot Logic ---
        if self.curriculum_stage == STAGE_DUEL:
            # Override Opponent (Pod 2) actions with Simple Bot
            # Bot: Steer towards next checkpoint
            
            # Opp is index 2
            # Get target
            opp_nid = self.next_cp_id[:, 2]
            batch_idx = torch.arange(self.num_envs, device=self.device)
            target = self.checkpoints[batch_idx, opp_nid]
            p_pos = self.physics.pos[:, 2]
            
            # Desired Angle
            diff = target - p_pos
            desired_rad = torch.atan2(diff[:, 1], diff[:, 0])
            desired_deg = torch.rad2deg(desired_rad)
            
            # --- Dynamic Difficulty Scaling ---
            # Difficulty 0.0: Thrust 60%, Steering Error +/- 15 deg
            # Difficulty 1.0: Thrust 100%, Steering Error 0 deg
            
            # 1. Steering Noise
            # Lower difficulty = More noise
            # Noise range: (1.0 - diff) * 30.0
            noise_scale = (1.0 - self.bot_difficulty) * 30.0
            noise = (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0) * noise_scale
            desired_deg += noise
            
            # Current Angle

            curr_deg = self.physics.angle[:, 2]
            
            # Delta
            delta = desired_deg - curr_deg
            # Normalize -180..180
            delta = (delta + 180) % 360 - 180
            
            # Clamp to -18..18
            delta = torch.clamp(delta, -18.0, 18.0)
            
            # Set Actions
            act_angle[:, 2] = delta
            
            # 2. Thrust Scaling
            # 40 + (60 * diff)
            thrust_val = 40.0 + (60.0 * self.bot_difficulty)
            act_thrust[:, 2] = thrust_val 
            
            act_shield[:, 2] = False
            act_boost[:, 2] = False
        
        # Track Previous Position for Continuous Collision Detection (CCD)
        prev_pos = self.physics.pos.clone()

        # 1. Physics Step
        collisions = self.physics.step(act_thrust, act_angle, act_shield, act_boost)
        
        # Update Roles based on new state (logic: "Assign it based on race state")
        # Do we update before or after rewards? 
        # Plan implies Role dictates Policy and Reward weights.
        # Usually update based on 'current' state before taking action?
        # But here valid state is post-physics.
        # Let's update roles NOW so rewards align with the *resulting* state?
        # Or roles from *previous* state used for observing are the ones that matter?
        # "The AI knows 'I am the Runner'". That was decided at obs time step t.
        # So we should use `self.is_runner` (computed at t) to calculate rewards for t->t+1?
        # Yes. The action was taken knowing "I am Runner", so reward should judge it as Runner.
        
        # So we use `self.is_runner` AS IS for rewards, THEN update it for next obs.

        
        # 2. Game Logic
        rewards = torch.zeros((self.num_envs, 2), device=self.device)

        # Metric Helpers
        infos = {
             "laps_completed": torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device),
             "checkpoints_passed": torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device),
             "current_streak": self.cps_passed.clone(),
             "cp_steps": torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device) # Steps taken for passed CPs
        }

        # --- Dense Rewards (Progress) ---
        new_dists = torch.zeros_like(self.prev_dist)
        
        for i in range(4):
            next_ids = self.next_cp_id[:, i]
            batch_indices = torch.arange(self.num_envs, device=self.device)
            target_pos = self.checkpoints[batch_indices, next_ids]
            curr_pos = self.physics.pos[:, i]
            new_dists[:, i] = torch.norm(target_pos - curr_pos, dim=1)
            
        if self.curriculum_stage == STAGE_DUEL:
             # Just ensure bot logic doesn't crash
             pass

        delta = self.prev_dist - new_dists
        # velocity weight is per agent/env.
        # w_velocity is [B]
        
        # SOTA: Potential-Based Reward Shaping
        # Reward = Phi(s') - Phi(s)
        # Here, Phi(s) = -DistanceToTarget
        # So Reward = (-NewDist) - (-PrevDist) = PrevDist - NewDist = delta
        # This guarantees optimal policy invariance compared to dense heuristics.
        
        weighted_delta = delta
        
        scale = w_velocity.unsqueeze(1) # [B, 1] broadcast to 4 pods?
        # rewards is [B, 2]. Delta is [B, 4].
        # We sum delta for pods 0+1 and 2+3.
        # Need to apply weight first.
        
        d_scaled = weighted_delta * scale
        
        
        # Annealing (tau)
        dense_mult = (1.0 - tau)
        # dense_mult is scalar or tensor? Assumed scalar for now.
        if isinstance(dense_mult, torch.Tensor):
             dense_mult = dense_mult.squeeze()
             
        rewards[:, 0] += (d_scaled[:, 0] + d_scaled[:, 1]) * dense_mult
        rewards[:, 1] += (d_scaled[:, 2] + d_scaled[:, 3]) * dense_mult

        # Orientation Reward (Guide to face next checkpoint)
        # ------------------------------------------------
        # 1. Calculate desired angle to next checkpoint
        # 2. Compare with current angle
        # 3. Reward alignment
        
        # We need vector to next CP for ALL 4 pods
        # We already computed 'new_dists' which uses 'target_pos'.
        # Let's re-gather target_pos or reuse if possible. 
        # For simplicity/clarity, re-gather or compute vectors.
        
        # Reuse loop for efficiency if possible? 
        # The loop above (lines 382-387) computes 'new_dists'.
        # We can just do a vectorized gather for all pods at once?
        # No, next_cp_id is [B, 4]. Checkpoints is [B, M, 2].
        
        
        # Orientation Reward
        orientation_rewards = torch.zeros((self.num_envs, 4), device=self.device)
        # w_orient is [B]
        
        # Optimization: if ALL w_orient are 0, skip.
        # Optimization: if ALL w_orient are 0, skip.
        if w_orient.sum() > 0.0:
            for i in range(4):
                next_ids = self.next_cp_id[:, i]
                batch_indices = torch.arange(self.num_envs, device=self.device)
                target_pos = self.checkpoints[batch_indices, next_ids] # [B, 2]
                curr_pos = self.physics.pos[:, i] # [B, 2]
                
                # Vector to target
                diff = target_pos - curr_pos
                # Angle of vector
                target_angle = torch.atan2(diff[:, 1], diff[:, 0]) # Radians
                
                # Current Angle
                curr_angle = torch.deg2rad(self.physics.angle[:, i]) # Radians
                
                # Alignment = Cos(Target - Current)
                alignment = torch.cos(target_angle - curr_angle)

                # --- Refined Orientation Logic ---
                # 1. Positive Reward: Narrow Cone (~60 deg)
                # Map [0.5, 1.0] -> [0.0, 1.0]
                THRESHOLD = 0.5
                pos_score = torch.clamp((alignment - THRESHOLD) / (1.0 - THRESHOLD), 0.0, 1.0)

                # 2. Negative Penalty: Wrong Way
                # If cos < 0, apply penalty weight.
                neg_score = torch.zeros_like(alignment)
                neg_mask = alignment < 0
                neg_score[neg_mask] = alignment[neg_mask] * w_wrong_way[neg_mask]

                # Combine
                # Note: w_orient scales the positive reward.
                # The negative penalty is already scaled by w_wrong_way.
                orientation_rewards[:, i] = (pos_score * w_orient) + neg_score
            
            # Mask inactive pods to prevent noise/bias
            # SOLO: Pod 0 active.
            # DUEL: Pod 0, 2 active.
            # LEAGUE: All active.
            if self.curriculum_stage == STAGE_SOLO:
                orientation_rewards[:, 1:] = 0.0
            elif self.curriculum_stage == STAGE_DUEL:
                orientation_rewards[:, 1] = 0.0
                orientation_rewards[:, 3] = 0.0

            # Add to rewards
            rewards[:, 0] += (orientation_rewards[:, 0] + orientation_rewards[:, 1]) * dense_mult
            rewards[:, 1] += (orientation_rewards[:, 2] + orientation_rewards[:, 3]) * dense_mult

        # Step Penalty (Discourage circling/loitering)
        # Apply to all
        # Progressive Penalty:
        # 0 penalty for first half of timeout.
        # Linearly scales to 1.0 * step_pen by end of timeout.
        
        # Step Penalty
        # w_step_pen [B]
        
        if w_step_pen.sum() > 0:
            # elapsed = TIMEOUT_STEPS - self.timeouts # [B, 4]
            # But self.timeouts was NOT decremented yet? 
            # self.timeouts starts at TIMEOUT_STEPS. 
            # We decrement at line 572. So currently it is the value for THIS step.
            # wait, step() is called for t -> t+1.
            # self.timeouts represents "steps remaining".
            
            steps_remaining = self.timeouts # [B, 4]
            total_steps = TIMEOUT_STEPS
            
            # Linear Penalty from Step 0
            # Alpha goes from 0.0 (at Start) to 1.0 (at Timeout) ? 
            # Or constant?
            # Standard RL: Constant penalty per step encourages speed.
            # "Alpha" approach was to panic them at the end.
            # Let's simple normalize: penalty = w_step_pen * (1.0) ?
            # No, let's keep the "Urgency" factor but make it start immediately.
            # alpha = (Total - Remaining) / Total
            # At Start (Rem=100, Total=100): Alpha = 0.
            # At End (Rem=0): Alpha = 1.
            # This means step 0 is free? We want to remove free buffer.
            # We want Alpha > 0 immediately?
            # actually, standard -0.1 per step is best.
            # Let's try: alpha = 1.0 always.
            # Then w_step_pen (15.0) is subtracted every step.
            # That's huge. 15 * 100 steps = -1500. Matches Win Reward.
            # This effectively puts a "Time Limit" cost.
            # Let's go with alpha = 1.0 (Constant Penalty).
            
            # Correction: User might prefer the "Panic" curve but without the zero-start.
            # Let's use alpha = 0.2 + 0.8 * (Progress)
            # Starts at 0.2, ends at 1.0.
            
            progress = (total_steps - steps_remaining) / total_steps
            alpha = 0.2 + (0.8 * progress)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            
            # Apply to team rewards
            # Team 0: Pods 0, 1
            # Team 1: Pods 2, 3
            
            # We average the penalty for the team? Or sum?
            # Original code:
            # rewards[:, 0] -= step_pen
            # that applied 'step_pen' ONCE per team per step?
            # Or is it per pod?
            # The original code just did `rewards -= step_pen`.
            # If we have 2 pods, and both are penalized, should we subtract 2 * step_pen?
            # The original code: `rewards[:, 0] -= step_pen`.
            # This implies a SINGLE penalty term per step for the team, regardless of pod count?
            # Or maybe it assumes implicit aggregation.
            # Let's Avg the alpha for the active pods of the team to maintain scale.
            
            # Active Masks
            # Solo: Pod 0 only.
            # Duel: Pod 0, 2 only.
            # League: All.
            
            active_mask = torch.zeros_like(steps_remaining, dtype=torch.bool)
            if self.curriculum_stage == STAGE_SOLO:
                active_mask[:, 0] = True
            elif self.curriculum_stage == STAGE_DUEL:
                active_mask[:, 0] = True
                active_mask[:, 2] = True
            else:
                active_mask[:] = True
                
            # Team 0 Alphas
            # alphas for pod 0, 1
            t0_alphas = alpha[:, 0:2]
            t0_mask = active_mask[:, 0:2]
            # Valid count
            t0_cnt = t0_mask.sum(dim=1).float() # [B]
            t0_cnt[t0_cnt == 0] = 1.0 # Avoid div/0
            
            t0_val = (t0_alphas * t0_mask.float()).sum(dim=1) / t0_cnt
            rewards[:, 0] -= w_step_pen * t0_val
            
            # Team 1 Alphas
            t1_alphas = alpha[:, 2:4]
            t1_mask = active_mask[:, 2:4]
            t1_cnt = t1_mask.sum(dim=1).float()
            t1_cnt[t1_cnt == 0] = 1.0
            
            t1_val = (t1_alphas * t1_mask.float()).sum(dim=1) / t1_cnt
            rewards[:, 1] -= w_step_pen * t1_val

        
        # Check Checkpoints
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
                self.timeouts[pass_idx, i] = TIMEOUT_STEPS 
                
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
                    if self.curriculum_stage == STAGE_SOLO:
                        # If Pod 0 finished a lap
                        if i == 0:
                             self.stage_metrics["solo_completes"] += len(z_idx)
                

                
                # --- Metric: Checkpoints Passed ---
                # Only count for active pod (Pod 0 in Solo)?
                # Let's count all active pods to be fair metric of "system activity".
                # But user wants "progress".
                if self.curriculum_stage == STAGE_SOLO:
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
                
                self.cps_passed[pass_idx, i] += 1
                infos["checkpoints_passed"][pass_idx, i] = 1
                streak = self.cps_passed[pass_idx, i].float() # 1, 2, 3...
                
                # Record Efficiency (Steps taken)
                taken_steps = self.steps_last_cp[pass_idx, i]
                infos["cp_steps"][pass_idx, i] = taken_steps
                self.steps_last_cp[pass_idx, i] = 0
                
                # --- TIME EXTENSION ---
                # Reset Timeout for this pod
                self.timeouts[pass_idx, i] = TIMEOUT_STEPS
                
                base_reward = w_checkpoint[pass_idx]
                scale_reward = w_chk_scale[pass_idx]
                
                # Reward = Base + (Streak-1)*Scale ? Or just Base + Streak*Scale?
                # "Giving more and more points"
                # If Streak=1 (First CP): Base (500)
                # If Streak=2: Base + Scale (1000)
                # Logic: total = base + (streak - 1) * scale
                
                total_reward = base_reward + (streak - 1.0) * scale_reward
                
                # Add to buffer for the specific POD
                self.cp_reward_buffer[pass_idx, i] += total_reward
                
                # Correction for Dense Reward Overshoot
                # If we passed, we reached the target (distance 0). 
                # The Dense Logic penalized us for moving "away" from it (the overshoot).
                # We add back the distance from target * scale to treat it as "Arrived at 0".
                # passed is bool mask [B]. pass_idx is indices.
                # new_dists is [B, 4].
                overshoot_dist = new_dists[pass_idx, i]
                
                # Overshoot correction should be immediate? 
                # Yes, it's a correction for THIS step's dense calculation.
                # correction for THIS step's dense calculation.
                team = i // 2
                rewards[pass_idx, team] += overshoot_dist * scale[pass_idx, 0] * dense_mult # scale is [B,1]
        
        # --- Payout Checkpoint Buffer ---
        # Spread over 10 steps
        PAYOUT_STEPS = 10.0
        # --- Payout Checkpoint Buffer ---
        # Spread over 10 steps
        PAYOUT_STEPS = 10.0
        payout_chunk = w_checkpoint / PAYOUT_STEPS # [B]
        payout_chunk = payout_chunk.unsqueeze(1) # [B, 1] for keying
        
        # Calculated actual payout (min of remaining or chunk)
        # Apply to all pods
        payout = torch.min(self.cp_reward_buffer, payout_chunk) # min works elementwise
        self.cp_reward_buffer -= payout
        
        # Add to Team Rewards
        # Team 0: 0, 1
        rewards[:, 0] += payout[:, 0] + payout[:, 1]
        rewards[:, 1] += payout[:, 2] + payout[:, 3]

        # Increment Step Counter for Efficiency
        self.steps_last_cp += 1

        # 3. Timeouts
        self.timeouts -= 1
        timed_out = (self.timeouts <= 0)
        
        # If any ACTIVE pod times out, the environment is done.
        # This prevents getting stuck in early training (Stage 0/1).
        env_timed_out = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        if self.curriculum_stage == STAGE_SOLO:
            # Only Pod 0 matters
            env_timed_out = timed_out[:, 0]
        elif self.curriculum_stage == STAGE_DUEL:
            # Pods 0 and 2 matter
            env_timed_out = timed_out[:, 0] | timed_out[:, 2]
        else:
            # Stage 2 (League): If any pod times out?
            env_timed_out = timed_out.any(dim=1)

        self.dones = self.dones | env_timed_out
        
        # 4. Win Condition
        # "First team to have a pod complete all laps wins."
        # Check laps
        finished = (self.laps >= MAX_LAPS)
        
        if finished.any():
            # Who finished?
            # finished is [B, 4]
            # Prioritize first found?
            # In sim, multiple could finish same step.
            # Mark winners.
            
            # Map back to env
            env_won = finished.any(dim=1)
            winner_indices = torch.nonzero(env_won).squeeze(-1)
            
            # Determine team
            # If both teams finish same frame? Draw/Tiebreaker.
            # Simple: Take first pod.
            # We need to set self.winners[env_idx] = team_id
            
            # We can iterate or use mask logic.
            # Mask for Team 0 finished: finished[:, 0] | finished[:, 1]
            # Mask for Team 1 finished: finished[:, 2] | finished[:, 3]
            
            t0_wins = (finished[:, 0] | finished[:, 1])
            t1_wins = (finished[:, 2] | finished[:, 3])
            
            # If both, random or draw. Let's say T0 wins (Host advantage).
            self.winners[t0_wins] = 0
            # Overwrite if T1 wins and T0 didn't?
            # logic: self.winners[t1_wins & ~t0_wins] = 1
            self.winners[t1_wins & ~t0_wins] = 1
            
            # Done
            self.dones[env_won] = True
            
            # Sparse Reward
            # Winner +1000, Loser -1000
            # Apply to `rewards`
            # For environments that are done:
            
            # w = self.winners[env_won] # [W] (0 or 1)
            # rewards[env_won, w] += 1000
            # rewards[env_won, 1-w] -= 1000
            
            # Vectorized assignment
            # Vectorized assignment
            mask_w0 = (self.winners == 0) & env_won
            mask_w1 = (self.winners == 1) & env_won
            
            rewards[mask_w0, 0] += w_win[mask_w0]
            rewards[mask_w0, 1] -= w_loss[mask_w0]
            
            rewards[mask_w1, 1] += w_win[mask_w1]
            rewards[mask_w1, 0] -= w_loss[mask_w1]
            
            # --- Curriculum Metric: Duel Win ---
            if self.curriculum_stage == STAGE_DUEL:
                # If T0 wins (mask_w0)
                n_wins = mask_w0.sum().item()
                n_games = env_won.sum().item()
                self.stage_metrics["duel_wins"] += n_wins
                self.stage_metrics["duel_games"] += n_games
                
                # Update Recent Metrics for Dynamic Difficulty
                self.stage_metrics["recent_wins"] += n_wins
                self.stage_metrics["recent_games"] += n_games

        # Update Progress Metric for Next Step
        self.update_progress_metric(torch.arange(self.num_envs, device=self.device))
        
        # --- Role Specific Collision Rewards ---
        # collisions: [B, 4, 4] magnitudes
        # Runner:
        #   - Avoid hitting Enemy (-0.5 * I)
        # Blocker:
        #   - Hit Enemy Runner (+2.0 * I)
        #   - Push Enemy Backwards (+5.0 * DeltaVel) (TODO: Need velocity delta from physics? Maybe Phase 4 optimization)
        
        for i in range(4):
            team = i // 2
            enemy_team = 1 - team
            enemy_indices = [2*enemy_team, 2*enemy_team + 1]
            
            # Am I Runner?
            is_run = self.is_runner[:, i] # [B]
            
            # My Collisions with Enemies
            # collisions[:, i, e1] + collisions[:, i, e2]
            impact_e1 = collisions[:, i, enemy_indices[0]]
            impact_e2 = collisions[:, i, enemy_indices[1]]
            total_impact = impact_e1 + impact_e2
            
            # 1. Runner Penalty
            # If I am runner, penalize impact
            runner_pen = -w_col_run * total_impact # w_col_run is [B]
            # Apply only if Runner
            # Add to TEAM reward
            self.rewards[:, team] += runner_pen * is_run.float()
            
            # 2. Blocker Bonus
            # If I am Blocker, Reward collision with ENEMY RUNNER.
            # Which enemy is runner?
            is_block = ~is_run
            
            enemy_runner_mask = self.is_runner[:, enemy_indices] # [B, 2]
            
            # We need to pick the impact corresponding to the enemy runner
            # impact_e1 corresponds to enemy_indices[0]
            # impact_e2 corresponds to enemy_indices[1]
            
            # e1_is_run: enemy_runner_mask[:, 0]
            # e2_is_run: enemy_runner_mask[:, 1]
            
            bonus = torch.zeros(self.num_envs, device=self.device)
            bonus += impact_e1 * enemy_runner_mask[:, 0].float()
            bonus += impact_e2 * enemy_runner_mask[:, 1].float()
            
            # Scale
            blocker_reward = w_col_block * bonus
            
            self.rewards[:, team] += blocker_reward * is_block.float()
            
        # Update Roles for Next Step
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._update_roles(all_ids)


        return rewards, self.dones.clone(), infos

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
        pad = torch.zeros_like(v_mag)
        
        # Assemble Self
        # [B, 4, 14]
        # v_local: 2, t_vec_l: 2, dest: 1, align: 2, shield, boost, timeout, lap, leader, v_mag, pad
        self_obs = torch.cat([
            v_local, t_vec_l, dest, align, shield, boost, timeout, lap, leader, v_mag, pad
        ], dim=-1)
        
        # --- Entity Features (3 x 13) ---
        # We need "Others" for each Pod.
        # Indices map:
        # 0 -> 1, 2, 3
        # 1 -> 0, 2, 3
        # 2 -> 0, 1, 3
        # 3 -> 0, 1, 2
        other_indices = torch.tensor([
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2]
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
        # entity_obs: [B, 4, 3, 13] -> [4, B, 3, 13]
        ent_c = entity_obs.permute(1, 0, 2, 3).contiguous()
        # cp_obs: [B, 4, 6] -> [4, B, 6]
        cp_c = cp_obs.permute(1, 0, 2).contiguous()
        
        return self_c, ent_c, cp_c
