import numpy as np

from training.evolution import fast_non_dominated_sort, calculate_crowding_distance

def test_nsga_sort_simple():
    # 2D Objective: Maximize X, Maximize Y
    # A: [10, 10] (Best)
    # B: [5, 5] (Dominated by A)
    # C: [10, 5] (Dominated by A? No. A=[10,10]. 10>=10 and 10>5. Yes Dominated.)
    # D: [5, 10] (Dominated by A)
    # E: [12, 2] (Non-dominated by A)
    
    # A vs E:
    # A: [10, 10], E: [12, 2]
    # A dominates E? 10 >= 12 False.
    # E dominates A? 12 >= 10 True. 2 >= 10 False.
    # So A and E are Front 0.
    
    objectives = np.array([
        [10, 10], # 0
        [5, 5],   # 1
        [10, 5],  # 2
        [5, 10],  # 3
        [12, 2]   # 4
    ])
    
    fronts = fast_non_dominated_sort(objectives)
    
    # Expected:
    # Front 0: [0, 4] (A, E)
    # Front 1: [2, 3] (C, D) -- C is [10, 5]. Dominated by A (10>=10, 10>5). 
    # But C vs E: [10, 5] vs [12, 2]. C not dom E (5>2). E not dom C (12>10).
    # So C is "Non-dominated with respect to E".
    # But A dominates C. So C cannot be Front 0.
    # So C is in Front 1 (dominated only by Front 0 members).
    
    # Let's trace C ([10, 5]):
    # Dominated by A ([10, 10]). Count=1.
    # Dominated by E ([12, 2])? No.
    # So Count=1. When A is processed (Front 0), C count -> 0. Added to Front 1. Correct.
    
    # F0: {0, 4}
    # F1: {2, 3} (C=[10,5], D=[5,10]. C vs D: 10>5, 5<10. Non-dom).
    # F2: {1} (B=[5,5]. Dominated by A, C, D. Count=3.)
    
    assert 0 in fronts[0]
    assert 4 in fronts[0]
    assert len(fronts[0]) == 2
    
    assert 2 in fronts[1]
    assert 3 in fronts[1]
    
    assert 1 in fronts[2]
    
    print("Sort Tests Passed!")

def test_crowding_distance():
    # Front:
    # A: [1, 10]
    # B: [2, 8]
    # C: [3, 6]
    # D: [4, 4]
    
    # Sorted by Obj 0: A, B, C, D
    # Sorted by Obj 1: D, C, B, A
    
    # Boundaries: A and D should have inf distance.
    
    objectives = np.array([
        [1.0, 10.0], # 0
        [2.0, 8.0],  # 1
        [3.0, 6.0],  # 2
        [4.0, 4.0]   # 3
    ])
    
    fronts = [[0, 1, 2, 3]]
    dist = calculate_crowding_distance(objectives, fronts)
    
    assert dist[0] == float('inf')
    assert dist[3] == float('inf')
    assert dist[1] > 0
    assert dist[2] > 0
    
    print("Crowding Tests Passed!")

if __name__ == "__main__":
    test_nsga_sort_simple()
    test_crowding_distance()
