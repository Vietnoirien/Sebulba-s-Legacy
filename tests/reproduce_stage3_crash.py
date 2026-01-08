import torch
import unittest

class Config:
    def __init__(self):
        self.pop_size = 32
        self.envs_per_agent = 128
        self.num_steps = 16
        self.lstm_hidden_size = 64
        self.seq_length = 8

class MockPPO:
    def __init__(self):
        self.config = Config()
        self.device = 'cpu'
        self.current_num_steps = self.config.num_steps
        
    def allocate_buffers(self, num_active_pods):
        active_pods = list(range(num_active_pods))
        self.current_active_pods_count = len(active_pods)
        
        num_active_per_agent_step = self.current_active_pods_count * self.config.envs_per_agent
        
        print(f"Allocating Buffers. Active Pods: {self.current_active_pods_count}. Per Agent: {num_active_per_agent_step}")
        
        self.agent_batches = []
        for _ in range(self.config.pop_size):
             self.agent_batches.append({
                 'actions': torch.zeros((self.current_num_steps, num_active_per_agent_step, 4), device=self.device)
             })
             
    def test_assignment(self, num_active_pods):
        self.allocate_buffers(num_active_pods)
        
        # Simulate v_act output from inference
        # Shape: [Pop, Batch, 4] where Batch = num_active_per_agent
        num_active_per_agent = num_active_pods * self.config.envs_per_agent
        
        # v_act shape logic from PPO:
        # v_act = fix_diag(v_act) -> [Pop, Batch, 4]
        # It comes from vmap of functional_inference which processes chunks.
        
        # Here we just mock the TENSOR causing the crash.
        # It's v_act[i].detach()
        
        v_act = torch.randn(self.config.pop_size, num_active_per_agent, 4)
        
        step = 0
        i = 0
        
        print(f"Testing Assignment. v_act[i] shape: {v_act[i].shape}")
        
        # The crash line:
        # batch['actions'][step] = v_act[i].detach().reshape(self.config.envs_per_agent * 1, 4) -- OLD
        # batch['actions'][step] = v_act[i].detach().reshape(num_active_per_agent, 4) -- NEW
        
        batch = self.agent_batches[i]
        
        try:
            # THIS IS THE FIXED LINE LOGIC
            batch['actions'][step] = v_act[i].detach().reshape(num_active_per_agent, 4)
            print("Successfully assigned actions with FIXED logic.")
        except RuntimeError as e:
            print(f"Caught RuntimeError with FXIED logic: {e}")
            raise e
            
        # Verify correctness of shape
        assert batch['actions'][step].shape == (num_active_per_agent, 4)
        
        # REPRODUCE CRASH check (Optional, to prove it would fail)
        if num_active_pods > 1:
            try:
                # This mimics the HARDCODED old logic
                bad_reshape = v_act[i].detach().reshape(self.config.envs_per_agent * 1, 4)
                print("Old logic Unexpectedly worked??")
            except RuntimeError as e:
                print(f"Confirmed Old Logic Fails: {e}")

class TestStage3Crash(unittest.TestCase):
    def test_stage3_shapes(self):
        ppo = MockPPO()
        
        # Case 1: Stage 2 (1 Pod) - Should pass both
        print("\n--- Test Case 1: 1 Active Pod ---")
        ppo.test_assignment(1)
        
        # Case 2: Stage 3 (2 Pods) - Should pass Fixed, Fail Old
        print("\n--- Test Case 2: 2 Active Pods ---")
        ppo.test_assignment(2)

if __name__ == '__main__':
    unittest.main()
