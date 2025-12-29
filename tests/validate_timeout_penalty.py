
def calculate_penalty(passed_cps, total_cps=18, rate=25.0, steps=100):
    max_penalty = steps * rate
    # Safety: Cannot divide by zero.
    if total_cps == 0: return max_penalty
    
    progress = passed_cps / total_cps
    # Clamp progress to [0, 1]
    progress = max(0.0, min(1.0, progress))
    
    penalty = max_penalty * (1.0 - progress)
    return penalty

print("--- Timeout Penalty Validation ---")
print(f"Scenario 1: Sitter (0 CPs)      -> Penalty: {calculate_penalty(0):.1f}")
print(f"Scenario 2: Beginner (3 CPs)    -> Penalty: {calculate_penalty(3):.1f}")
print(f"Scenario 3: Mid-Race (9 CPs)    -> Penalty: {calculate_penalty(9):.1f}")
print(f"Scenario 4: Near End (17 CPs)   -> Penalty: {calculate_penalty(17):.1f}")
print(f"Scenario 5: Finished (18 CPs)   -> Penalty: {calculate_penalty(18):.1f}")
print("--------------------------------")
