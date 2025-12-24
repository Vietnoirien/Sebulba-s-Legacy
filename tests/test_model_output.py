import torch
from models.deepsets import PodAgent

def test_model_stability():
    agent = PodAgent()
    
    # Simulate large inputs (exploding observation)
    B = 10
    self_obs = torch.randn(B, 14) * 1000.0
    entity_obs = torch.randn(B, 3, 13) * 1000.0
    next_cp_obs = torch.randn(B, 6) * 1000.0
    
    print("Testing with large inputs...")
    try:
        action, log_prob, entropy, value = agent.get_action_and_value(self_obs, entity_obs, next_cp_obs)
        print("Success! Output received.")
        print(f"Action mean: {action.mean().item()}")
        print(f"Action max: {action.max().item()}")
        print(f"Action min: {action.min().item()}")
        
        # Check internal checks
        # We can't easily check 'means' without hacking the class or using a hook, 
        # but if get_action_and_value returns without erroring on Normal(), we are good.
        
    except ValueError as e:
        print(f"Caught ValueError: {e}")
        exit(1)
    except Exception as e:
        print(f"Caught Unexpected Error: {e}")
        exit(1)

if __name__ == "__main__":
    test_model_stability()
