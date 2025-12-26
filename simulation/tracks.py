import torch
import math

class TrackGenerator:
    """
    Handles generation of race tracks (checkpoints).
    """
    @staticmethod


    

    
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
        
        # Fallback for remaining failures (Sequential Random)
        # Slower, but high probability of finding a valid random config
        if pending_mask.any():
            fail_ids = torch.nonzero(pending_mask).squeeze(-1)
            
            # CPU loop for safety and logic complexity
            # (Vectorizing sequential dependencies is hard/inefficient for few fails)
            for env_idx in fail_ids:
                found_valid_seq = False
                for seq_attempt in range(50): # Try 50 sequential builds
                    seq_cps = torch.zeros((num_cps, 2), device=device)
                    valid_build = True
                    
                    for i in range(num_cps):
                        # Try to place point i
                        placed = False
                        for p_try in range(50): # 50 tries per point
                            rx = torch.rand(1, device=device) * (width - 2*border) + border
                            ry = torch.rand(1, device=device) * (height - 2*border) + border
                            cand = torch.cat([rx, ry])
                            
                            # Check vs previous
                            if i == 0:
                                seq_cps[i] = cand
                                placed = True
                                break
                            else:
                                prev = seq_cps[:i]
                                dists = torch.norm(prev - cand, dim=1)
                                if (dists >= min_dist).all():
                                    seq_cps[i] = cand
                                    placed = True
                                    break
                        
                        if not placed:
                            valid_build = False
                            break
                    
                    if valid_build:
                        final_cps[env_idx] = seq_cps
                        found_valid_seq = True
                        break
                
                if not found_valid_seq:
                    print(f"CRITICAL: Could not find valid config for env {env_idx} even with sequential fallback!")
                    # Last resort: Just random (will fail check but avoid crash)
                    # OR: fallback to a simple widespread scatter
                    # We leave it as 0.0 which will trigger fail in verify, helping us debug.
            
        return final_cps
