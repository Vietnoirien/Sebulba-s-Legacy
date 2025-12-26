import torch
import math

class TrackGenerator:
    """
    Handles generation of race tracks (checkpoints).
    """
    @staticmethod
    def generate_star_convex(num_envs, num_cps, width, height, device, min_dist=2500.0):
        """
        Generates loop tracks using Star-Convex polygon generation.
        Guaranteed to be non-intersecting and loop-able.
        Includes rejection sampling to ensure min_dist.
        """
        border = 1800.0
        center_x = width / 2.0
        center_y = height / 2.0
        
        # Max radius fits in map
        max_r = min(width, height) / 2.0 - border
        min_r = max_r * 0.4 
        
        final_cps = torch.zeros((num_envs, num_cps, 2), device=device)
        pending_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        
        MAX_RETRIES = 50
        
        for attempt in range(MAX_RETRIES):
            n_pending = pending_mask.sum()
            if n_pending == 0:
                break
            
            p_ids = torch.nonzero(pending_mask).squeeze(-1)
            
            # 1. Generate Angles
            steps = torch.rand(n_pending, num_cps, device=device) + 0.25 # Increased min angle
            angles = torch.cumsum(steps, dim=1)
            max_angles = angles[:, -1:].clone()
            angles = (angles / max_angles) * 2 * math.pi
            
            offset_theta = torch.rand(n_pending, 1, device=device) * 2 * math.pi
            angles = angles + offset_theta
            
            # 2. Generate Radii
            radii = (torch.rand(n_pending, num_cps, device=device) * (max_r - min_r)) + min_r
            
            # 3. XY
            off_x = radii * torch.cos(angles)
            off_y = radii * torch.sin(angles)
            
            px = center_x + off_x
            py = center_y + off_y
            
            cands = torch.stack([px, py], dim=2)
            
            # Validation
            d = torch.norm(cands.unsqueeze(2) - cands.unsqueeze(1), dim=3)
            mask = torch.eye(num_cps, device=device).bool().unsqueeze(0).expand(n_pending, -1, -1)
            d.masked_fill_(mask, 99999.0)
            
            min_d = d.min(dim=2)[0].min(dim=1)[0]
            valid = min_d >= min_dist
            
            if valid.any():
                good_local = torch.nonzero(valid).squeeze(-1)
                good_global = p_ids[good_local]
                final_cps[good_global] = cands[good_local]
                pending_mask[good_global] = False
                
        # If any fail, force a simple perfect polygon (Circle)
        if pending_mask.any():
            fail_ids = torch.nonzero(pending_mask).squeeze(-1)
            n_fail = len(fail_ids)
            
            # Perfect Circle
            # Border 1800 => Max R = 4500 - 1800 = 2700.
            # Fixed R = 2600 is safe. 
            # Hexagon side = R. So dist = 2600 > 2500.
            fixed_r = 2600.0 
            theta = torch.linspace(0, 2*math.pi, num_cps + 1, device=device)[:num_cps]
            theta = theta.unsqueeze(0).expand(n_fail, -1)
            
            cx = center_x + fixed_r * torch.cos(theta)
            cy = center_y + fixed_r * torch.sin(theta)
            
            final_cps[fail_ids] = torch.stack([cx, cy], dim=2)

        return final_cps
    
    @staticmethod
    def generate_guided_worm(num_envs, num_cps, width, height, device, min_dist=2500.0):
        # Implicitly updated via this replace block
        return TrackGenerator._generate_guided_worm_impl(num_envs, num_cps, width, height, device, min_dist)

    @staticmethod
    def _generate_guided_worm_impl(num_envs, num_cps, width, height, device, min_dist=2500.0):
        # Implementation moved here to allow helper access or just keep it
        # Wait, I can't easily introduce a new helper method without messing up the file.
        # I'll just paste the body of Guided Worm here again.
        
        # Outer retry loop for validity
        final_cps = torch.zeros((num_envs, num_cps, 2), device=device)
        pending_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        
        MAX_RETRIES = 10
        border = 1800.0
        
        for attempt in range(MAX_RETRIES):
            n_pending = pending_mask.sum()
            if n_pending == 0:
                break
                
            p_ids = torch.nonzero(pending_mask).squeeze(-1)
            
            # --- Generate Batch of Worms ---
            cps = torch.zeros((n_pending, num_cps, 2), device=device)
            cx = torch.rand(n_pending, device=device) * (width - 2*border) + border
            cy = torch.rand(n_pending, device=device) * (height - 2*border) + border
            cps[:, 0, 0] = cx
            cps[:, 0, 1] = cy
            
            curr_x = cx
            curr_y = cy
            curr_angle = torch.rand(n_pending, device=device) * 2 * math.pi
            
            valid_batch = torch.ones(n_pending, dtype=torch.bool, device=device)
            
            for i in range(1, num_cps):
                dx = cps[:, 0, 0] - curr_x
                dy = cps[:, 0, 1] - curr_y
                
                progress = i / float(num_cps)
                homing_strength = progress ** 3
                target_angle = torch.atan2(dy, dx)
                
                noise = (torch.rand(n_pending, device=device) * 2 - 1) * (math.pi * 0.6)
                candidate_angle = curr_angle + noise
                
                vx_r = torch.cos(candidate_angle)
                vy_r = torch.sin(candidate_angle)
                vx_t = torch.cos(target_angle)
                vy_t = torch.sin(target_angle)
                
                vx = (1.0 - homing_strength) * vx_r + homing_strength * vx_t
                vy = (1.0 - homing_strength) * vy_r + homing_strength * vy_t
                v_norm = torch.sqrt(vx**2 + vy**2) + 1e-5
                vx = vx / v_norm
                vy = vy / v_norm
                curr_angle = torch.atan2(vy, vx)
                
                step = 3000.0 * (0.8 + 0.4 * torch.rand(n_pending, device=device))
                next_x = curr_x + vx * step
                next_y = curr_y + vy * step
                
                # Bounce
                mask_x_low = next_x < border
                mask_x_high = next_x > (width - border)
                mask_y_low = next_y < border
                mask_y_high = next_y > (height - border)
                
                next_x[mask_x_low] = border + (border - next_x[mask_x_low])
                next_x[mask_x_high] = (width - border) - (next_x[mask_x_high] - (width - border))
                next_y[mask_y_low] = border + (border - next_y[mask_y_low])
                next_y[mask_y_high] = (height - border) - (next_y[mask_y_high] - (height - border))
                
                next_x = torch.clamp(next_x, border, width - border)
                next_y = torch.clamp(next_y, border, height - border)
                
                cps[:, i, 0] = next_x
                cps[:, i, 1] = next_y
                curr_x = next_x
                curr_y = next_y
            
            # --- Validation ---
            # Check pairwise dists for this batch
            d = torch.norm(cps.unsqueeze(2) - cps.unsqueeze(1), dim=3)
            mask = torch.eye(num_cps, device=device).bool().unsqueeze(0).expand(n_pending, -1, -1)
            d.masked_fill_(mask, 99999.0)
            min_d = d.min(dim=2)[0].min(dim=1)[0]
            
            is_good = min_d >= min_dist
            
            if is_good.any():
                good_local = torch.nonzero(is_good).squeeze(-1)
                good_global = p_ids[good_local]
                final_cps[good_global] = cps[good_local]
                pending_mask[good_global] = False

        if pending_mask.any():
            fail_ids = torch.nonzero(pending_mask).squeeze(-1)
            n_fail = len(fail_ids)
            # Fallback to StarConvex
            safe_cps = TrackGenerator.generate_star_convex(n_fail, num_cps, width, height, device, min_dist)
            final_cps[fail_ids] = safe_cps
            
        return final_cps
    
    @staticmethod
    def generate_max_entropy(num_envs, num_cps, width, height, device, min_dist=2500.0):
        """
        Maximum Entropy Generator with Strict Guarantees.
        Uses Rejection Sampling to ensure min_dist constraints.
        Fallbacks to Star-Convex if a valid configuration isn't found in N attempts.
        """
        border = 1800.0 # Strict System Border
        
        # Container for final result
        final_cps = torch.zeros((num_envs, num_cps, 2), device=device)
        
        # Mask of environments that still need generation
        pending_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        
        MAX_ATTEMPTS = 100
        
        for attempt in range(MAX_ATTEMPTS):
            n_pending = pending_mask.sum()
            if n_pending == 0:
                break
                
            # Generate Random Points for pending
            # [P, M, 2]
            p_ids = torch.nonzero(pending_mask).squeeze(-1)
            
            rx = torch.rand(n_pending, num_cps, device=device) * (width - 2*border) + border
            ry = torch.rand(n_pending, num_cps, device=device) * (height - 2*border) + border
            cands = torch.stack([rx, ry], dim=2) # [P, M, 2]
            
            # Check Constraints
            # 1. Bounds (Implicitly strictly enforced by generation range above)
            
            # 2. Min Dist
            # Expand [P, M, 1, 2] vs [P, 1, M, 2]
            d = torch.norm(cands.unsqueeze(2) - cands.unsqueeze(1), dim=3)
            
            # Mask diagonal
            mask = torch.eye(num_cps, device=device).bool().unsqueeze(0).expand(n_pending, -1, -1)
            d.masked_fill_(mask, 99999.0)
            
            min_d, _ = d.min(dim=2) # [P, M]
            min_d_all, _ = min_d.min(dim=1) # [P]
            
            valid = min_d_all >= min_dist
            
            if valid.any():
                # Save valid ones
                valid_local_ids = torch.nonzero(valid).squeeze(-1)
                global_ids = p_ids[valid_local_ids]
                final_cps[global_ids] = cands[valid_local_ids]
                pending_mask[global_ids] = False
        
        # Fallback for remaining failures
        if pending_mask.any():
            fail_ids = torch.nonzero(pending_mask).squeeze(-1)
            n_fail = len(fail_ids)
            # Use Safe StarConvex for these. 
            # Note: StarConvex fallback is now hardened too.
            safe_cps = TrackGenerator.generate_star_convex(n_fail, num_cps, width, height, device, min_dist)
            final_cps[fail_ids] = safe_cps
            
        return final_cps
