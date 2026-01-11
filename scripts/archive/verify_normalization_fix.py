import torch
from training.normalization import RunningMeanStd
# Mock PPOTrainer minimal
class MockPPO:
    def __init__(self):
        self.device = 'cpu'
        # Size 14 (Excluded 1)
        self.rms_self = RunningMeanStd((14,), device=self.device)
        
    def _normalize_self_selective(self, raw_self, fixed=False):
        """
        Normalizes self_obs (15 dims) while EXCLUDING Index 11 (Leader/is_runner).
        """
        # Split: Indices 0-10, 12-14 (Norm) | 11 (Pass)
        p1 = raw_self[..., :11]
        p2 = raw_self[..., 12:]
        leader = raw_self[..., 11:12]
        
        to_norm = torch.cat([p1, p2], dim=-1) # [..., 14]
        normed = self.rms_self(to_norm, fixed=fixed)
        
        n1 = normed[..., :11]
        n2 = normed[..., 11:]
        
        return torch.cat([n1, leader, n2], dim=-1) # [..., 15]

def test_norm():
    ppo = MockPPO()
    
    # Create Raw Input [B, 15]
    # Index 11 is Leader (0 or 1)
    # Other indices random
    B = 10
    raw = torch.randn(B, 15)
    
    # Set Leader explicitly
    raw[:, 11] = 1.0 # Runners
    raw[5:, 11] = 0.0 # Blockers
    
    # Run Norm
    normed = ppo._normalize_self_selective(raw)
    
    print("Normalizing...")
    
    # Check Index 11
    # Should be EXACTLY 1.0 or 0.0
    err = (normed[:, 11] - raw[:, 11]).abs().sum()
    
    if err < 1e-6:
        print("[PASS] Leader/Role Flag preserved exactly.")
    else:
        print(f"[FAIL] Leader Flag modified! err={err}")
        print("Raw:", raw[:, 11])
        print("Norm:", normed[:, 11])
        
    # Check others are changed (roughly)
    diff = (normed[:, 0] - raw[:, 0]).abs().sum()
    if diff > 1e-6:
         print(f"[PASS] Other features normalized (diff={diff:.4f}).")
    else:
         print(f"[FAIL] Other features NOT normalized (diff={diff}).")

if __name__ == "__main__":
    test_norm()
