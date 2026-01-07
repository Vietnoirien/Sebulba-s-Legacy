import torch
import math
from config import *

class GPUPhysics:
    def __init__(self, num_envs, device='cuda'):
        print(f"DEBUG: GPUPhysics init. num_envs type: {type(num_envs)}, value: {num_envs}, device type: {type(device)}")
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # State [Batch, 4, 2]
        self.pos = torch.zeros((num_envs, 4, 2), device=self.device)
        self.vel = torch.zeros((num_envs, 4, 2), device=self.device)
        self.angle = torch.zeros((num_envs, 4), device=self.device) # Degrees
        self.mass = torch.ones((num_envs, 4), device=self.device)
        self.shield_cd = torch.zeros((num_envs, 4), dtype=torch.int32, device=self.device)
        self.boost_available = torch.ones((num_envs, 2), dtype=torch.bool, device=self.device) # Per team
        
        # Define pairs [6, 2]. 0-indexed pods 0,1,2,3
        self.pairs = torch.tensor([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3],
            [2, 3]
        ], device=self.device, dtype=torch.long)
        
    def step(self, actions_thrust, actions_angle_offset, actions_shield, actions_boost):
        """
        actions_thrust: [Batch, 4]  (0 to 100)
        actions_angle_offset: [Batch, 4] (-18 to 18)
        actions_shield: [Batch, 4] (bool)
        actions_boost: [Batch, 4] (bool)
        """
        
        # 1. Rotate
        # Clamp offset
        angle_offset = torch.clamp(actions_angle_offset, -MAX_TURN_DEGREES, MAX_TURN_DEGREES)
        self.angle = self.angle + angle_offset
        # Normalize angle? cpu logic does it.
        # self.angle = (self.angle + 180) % 360 - 180  ... simplified
        
        # 2. Thrust
        # Handle Boost & Shield Mass
        # Logic: Shield sets mass=10.
        # Check shield activation
        # If shield active (cooldown > 0 or just activated), mass = 10?
        # Plan 1.3: "Increases mass x10 for 1 turn... Prevents accelerating for next 3 turns"
        
        # Reset mass
        self.mass.fill_(1.0)
        
        # Activate Shield
        can_shield = (self.shield_cd == 0)
        do_shield = actions_shield & can_shield
        self.shield_cd[do_shield] = 4 # 3 turns + current? Reference said 3+1
        
        # Apply Thrust
        # If shielding, thrust = 0? 
        # Plan 1.3: "Prevents accelerating".
        # If cd > 0, thrust = 0.
        is_shielding = (self.shield_cd > 0)
        self.mass[is_shielding] = SHIELD_MASS
        
        # effective thrust
        thrust = actions_thrust.clone()
        thrust[is_shielding] = 0.0

        # Activate Boost
        # Map pods to teams (0,1 -> 0; 2,3 -> 1)
        # self.boost_available is [Batch, 2]
        
        # Determine who CAN boost
        # Gather availability: [Batch, 2] -> [Batch, 4]
        team_indices = torch.tensor([0, 0, 1, 1], device=self.device).expand(self.num_envs, 4)
        available_expanded = self.boost_available.gather(1, team_indices)
        
        # Determine who IS boosting
        do_boost = actions_boost & available_expanded
        
        # Apply Thrust Override
        # If shielding, thrust is 0 (handled above/below).
        # Boost overrides shield? Usually Boost > Shield? 
        # Plan 1.3: "Shield... Prevents accelerating". Boost IS acceleration.
        # Logical priority: Shield blocks standard thrust. Does it block Boost?
        # Usually checking "Actions" -> Boost Action.
        # If I shield AND boost?
        # In typical games, Shield prevents all engine output.
        # Let's assume Shield blocks Boost too.
        do_boost = do_boost & (~is_shielding)
        
        # Apply Boost Thrust (650.0)
        thrust[do_boost] = BOOST_THRUST
        
        # Consume Boost (Team Resource)
        if do_boost.any():
            # Update team availability
            # If Pod 0 OR Pod 1 boosted, Team 0 used it.
            used_t0 = do_boost[:, 0] | do_boost[:, 1]
            used_t1 = do_boost[:, 2] | do_boost[:, 3]
            
            # Use masked write
            self.boost_available[used_t0, 0] = False
            self.boost_available[used_t1, 1] = False
        
        angle_rad = torch.deg2rad(self.angle)
        self.vel[..., 0] += thrust * torch.cos(angle_rad)
        self.vel[..., 1] += thrust * torch.sin(angle_rad)
        
        # 3. Move
        self.pos += self.vel
        
        # 4. Resolve Collisions
        collisions = self._resolve_collisions()
        
        # 5. Friction
        self.vel *= FRICTION
        
        # Clamp to MAX_SPEED
        # speed: [Batch, 4, 1]
        speed = torch.norm(self.vel, dim=-1, keepdim=True)
        is_over = speed > MAX_SPEED
        
        # is_over is [Batch, 4, 1]. We need [Batch, 4] for boolean indexing of [Batch, 4, 2]
        mask = is_over.squeeze(-1)
        
        if mask.any():
            # [K, 2] = ([K, 2] / [K, 1]) * Scalar
            self.vel[mask] = (self.vel[mask] / speed[mask]) * MAX_SPEED
            
        self.vel = torch.trunc(self.vel)
        
        # 6. Round
        self.pos = torch.round(self.pos)
        
        # Update counters
        self.shield_cd = torch.clamp(self.shield_cd - 1, min=0)
        
        return collisions

    def _resolve_collisions(self):
        K = 4
        # [Batch, 6, 2] indices
        idx_p1 = self.pairs[:, 0] # [6]
        idx_p2 = self.pairs[:, 1] # [6]
        
        # Track impacts: [Batch, 4, 4]
        # matrix[b, i, j] = accumulated impulse magnitude between i and j
        impacts = torch.zeros((self.num_envs, 4, 4), device=self.device)
        
        for _ in range(K):
            # Gather P1, P2 [Batch, 6, 2]
            p1_pos = self.pos[:, idx_p1]
            p2_pos = self.pos[:, idx_p2]
            p1_vel = self.vel[:, idx_p1]
            p2_vel = self.vel[:, idx_p2]
            m1 = self.mass[:, idx_p1].unsqueeze(-1) # [Batch, 6, 1]
            m2 = self.mass[:, idx_p2].unsqueeze(-1)
            
            # Diff
            delta_pos = p2_pos - p1_pos
            dist_sq = (delta_pos ** 2).sum(dim=-1) # [Batch, 6]
            
            # Mask
            min_dist = POD_RADIUS * 2
            mask_coll = dist_sq < (min_dist ** 2) # [Batch, 6]
            
            # Safe Dist
            dist = torch.sqrt(dist_sq)
            # Avoid div zero
            dist = torch.clamp(dist, min=0.001)
            
            # Normal
            nx = delta_pos[..., 0] / dist
            ny = delta_pos[..., 1] / dist
            # N = [Batch, 6, 2]
            n_vec = torch.stack([nx, ny], dim=-1)
            
            # Relative Vel
            rel_vel = p1_vel - p2_vel
            
            # Impact Force
            # v_rel . N
            prod = (rel_vel * n_vec).sum(dim=-1) # [Batch, 6]
            
            inv_m1 = 1.0 / m1.squeeze(-1)
            inv_m2 = 1.0 / m2.squeeze(-1)
            
            # F = prod / (inv_m1 + inv_m2)
            # Force should only be applied if prod > 0 (moving towards each other)? 
            # In elastic collision, yes.
            f = prod / (inv_m1 + inv_m2)
            
            # Min Impulse
            f = torch.clamp(f, min=MIN_IMPULSE)
            
            # Impulse J
            # j = -n * f
            j = -n_vec * f.unsqueeze(-1) # [Batch, 6, 2]
            
            # Apply only where colliding
            j = j * mask_coll.unsqueeze(-1).float()
            
            # Record Impacts
            # f is magnitude [Batch, 6]. Masked by mask_coll.
            # We want to store f in impacts[b, p1, p2]
            
            curr_impact = f * mask_coll.float() # [Batch, 6]
            
            # We loop k=6 to scatter this back to matrix
            for k in range(6):
                i1 = idx_p1[k]
                i2 = idx_p2[k]
                val = curr_impact[:, k]
                impacts[:, i1, i2] += val
                impacts[:, i2, i1] += val
            
            # Separation
            overlap = min_dist - dist
            sep = overlap / 2.0
            sep_vec = n_vec * sep.unsqueeze(-1)
            sep_vec = sep_vec * mask_coll.unsqueeze(-1).float()
            
            # Scatter Add
            # We need to sum up impulses/separations for each pod index
            
            # dVel = J / m
            dv1 = (j / m1) 
            dv2 = (-j / m2) 
            
            # dPos (Separation)
            # P1 moves -sep_vec (away from P2)
            # P2 moves +sep_vec
            dp1 = -sep_vec
            dp2 = sep_vec

            # Accumulate changes
            d_vel = torch.zeros_like(self.vel)
            d_pos = torch.zeros_like(self.pos)
            
            for k in range(6):
                idx1 = idx_p1[k]
                idx2 = idx_p2[k]
                
                # dv1[:, k, :] is [Batch, 2]
                d_vel[:, idx1] += dv1[:, k]
                d_vel[:, idx2] += dv2[:, k]
                
                d_pos[:, idx1] += dp1[:, k]
                d_pos[:, idx2] += dp2[:, k]
                
            self.vel += d_vel
            self.pos += d_pos
            
        return impacts
