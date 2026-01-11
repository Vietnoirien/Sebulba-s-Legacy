import torch

def reproduce():
    print("--- Reproducing Sign Error ---")
    
    # Case 1: BoolTensor (Correct)
    is_runner_bool = torch.tensor([True, False])
    is_blocker_bool = ~is_runner_bool
    print(f"Bool Input: {is_runner_bool}")
    print(f"Bool Output (~): {is_blocker_bool}")
    print(f"Float Output: {is_blocker_bool.float()}")
    # Expected: [0., 1.] for [True, False] input (since ~True=False=0, ~False=True=1)
    # Wait: ~True is False. ~False is True.
    # So is_runner=[1,0] -> is_blocker=[0,1]. Correct.
    
    # Case 2: FloatTensor (What user might have if initialized via ones/zeros)
    try:
        is_runner_float = torch.tensor([1.0, 0.0])
        # is_blocker_float = ~is_runner_float # This throws error usually
        # But if it works...
        pass
    except Exception as e:
        print(f"Float Tensor ~ Error: {e}")
        
    # Case 3: IntTensor (What user might have if initialized via randint or logic)
    is_runner_int = torch.tensor([1, 0], dtype=torch.int64)
    is_blocker_int = ~is_runner_int
    print(f"\nInt Input: {is_runner_int}")
    print(f"Int Output (~): {is_blocker_int}")
    print(f"Float Output: {is_blocker_int.float()}")
    
    penalty = -200.0
    reward_blocking = penalty * is_blocker_int.float()
    print(f"\nApplied Penalty (-200 * mask): {reward_blocking}")
    
    if (reward_blocking > 0).any():
        print("!!! CONFIRMED: Penalty became positive reward due to Int bitwise NOT !!!")
    else:
        print("Not reproduced with Int.")

if __name__ == "__main__":
    reproduce()
