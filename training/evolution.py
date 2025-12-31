import torch
import numpy as np
from typing import List, Dict, Tuple

def calculate_novelty(population: List[Dict], k: int = 5) -> List[float]:
    """
    Calculates sparsity-based novelty score for each agent.
    Novelty = Average distance to k-nearest neighbors in behavior space.
    Behavior vector must be in p['behavior'] (shape [D]).
    """
    if len(population) < 2:
        return [0.0] * len(population)
        
    # Stack behavior vectors
    # Ensure they are cpu tensors or numpy
    vecs = []
    for p in population:
        b = p.get('behavior', torch.zeros(2))
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        vecs.append(b)
        
    X = np.stack(vecs) # [Pop, D]
    
    # Normalize X to [0,1] to treat dimensions equally (Speed vs Steering)
    # Avoid div/0
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    range_val = max_val - min_val
    range_val[range_val < 1e-6] = 1.0
    
    X_norm = (X - min_val) / range_val
    
    n_pop = len(population)
    novelty_scores = []
    
    # Brute force distance matrix (Pop is small, ~32-64, so efficient enough)
    # Dist[i, j]
    for i in range(n_pop):
        dist = np.linalg.norm(X_norm - X_norm[i], axis=1)
        # Sort distances
        sorted_dist = np.sort(dist)
        
        # Take k nearest (excluding self at index 0)
        # sorted_dist[0] is 0.0 (self)
        # Neighbors are 1..k+1
        k_eff = min(k, n_pop - 1)
        if k_eff <= 0:
            score = 0.0
        else:
            knn_dists = sorted_dist[1 : k_eff + 1]
            score = np.mean(knn_dists)
            
        novelty_scores.append(float(score))
        
    return novelty_scores

def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    NSGA-II Fast Non-Dominated Sort.
    objectives: [N, M] array where N is population size, M is metrics.
    Assumes HIGHER is BETTER for all objectives.
    Returns: List of Fronts (List of indices). Front 0 is best.
    """
    n_pop = objectives.shape[0]
    
    dominates_list = [[] for _ in range(n_pop)] # S_p
    dominated_count = np.zeros(n_pop, dtype=int) # n_p
    rank = np.zeros(n_pop, dtype=int)
    
    fronts = [[]] # F_i
    
    for p in range(n_pop):
        for q in range(n_pop):
            if p == q: continue
            
            # Check p dominates q
            # Higher is better
            # p dominates q if p >= q in all AND p > q in at least one
            p_vals = objectives[p]
            q_vals = objectives[q]
            
            p_ge_q = np.all(p_vals >= q_vals)
            p_gt_q = np.any(p_vals > q_vals)
            
            if p_ge_q and p_gt_q:
                dominates_list[p].append(q)
            elif np.all(q_vals >= p_vals) and np.any(q_vals > p_vals):
                dominated_count[p] += 1
                
        if dominated_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break
            
    return fronts

def calculate_crowding_distance(objectives: np.ndarray, fronts: List[List[int]]) -> np.ndarray:
    """
    Calculate crowding distance for NSGA-II.
    objectives: [N, M]
    fronts: List of indices
    Returns: [N] distance array
    """
    n_pop, n_obj = objectives.shape
    distance = np.zeros(n_pop)
    
    for front in fronts:
        if len(front) == 0: continue
        
        # Set infinite distance to bondary points
        # Iterate per objective
        for m in range(n_obj):
            # Sort front by objective m
            # We want Higher = 0, Lower = Max? 
            # Crowding dist is relative to neighbors in sorted list
            sorted_front = sorted(front, key=lambda x: objectives[x, m])
            
            distance[sorted_front[0]] = np.inf
            distance[sorted_front[-1]] = np.inf
            
            m_range = objectives[sorted_front[-1], m] - objectives[sorted_front[0], m]
            if m_range == 0: m_range = 1e-6
            
            for i in range(1, len(sorted_front) - 1):
                prev_id = sorted_front[i-1]
                next_id = sorted_front[i+1]
                curr_id = sorted_front[i]
                
                # dist += (next - prev) / range
                d = (objectives[next_id, m] - objectives[prev_id, m]) / m_range
                distance[curr_id] += d
                
    return distance

def lexicographic_sort(population: List[Dict], stage: int) -> List[List[int]]:
    """
    Sorts population strictly based on hierarchical objectives.
    Returns a list of lists (fronts), where each list contains one index.
    This effectively assigns a unique rank 0..N to each agent.
    """
    # Helper for sort key
    def get_key(p):
        # Handle None values safely
        wins = p.get('ema_wins', 0.0)
        if wins is None: wins = 0.0
        
        cons = p.get('ema_consistency', 0.0)
        if cons is None: cons = 0.0
        
        eff = p.get('ema_efficiency', 999.0)
        if eff is None: eff = 999.0
        
        nov = p.get('novelty_score', 0.0)
        if nov is None: nov = 0.0
        
        nur = p.get('nursery_score', 0.0)
        if nur is None: nur = 0.0
        
        # STAGE Switch
        if stage == 0: # Nursery
            # 1. Consistency, 2. Nursery Score, 3. Novelty
            return (cons, nur, nov)
        else: # Solo (and potential fallback)
            # 1. Wins, 2. Consistency, 3. -Efficiency, 4. Novelty
            return (wins, cons, -eff, nov)

    # Sort population indices
    # We want DESCENDING order for the key (Higher is better)
    indices = list(range(len(population)))
    indices.sort(key=lambda i: get_key(population[i]), reverse=True)
    
    # Return as list of lists to mimic fronts: [[best], [2nd], ... [worst]]
    # This assigns Rank 0 to best, Rank 1 to 2nd best, etc.
    return [[i] for i in indices]
