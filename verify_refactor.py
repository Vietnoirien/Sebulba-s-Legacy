
try:
    from simulation.env import PodRacerEnv
    from training.ppo import PPOTrainer
    print("Imports successful.")
    trainer = PPOTrainer()
    print(f"Trainer created. Stage: {trainer.curriculum.current_stage_id}")
    trainer.env.reset()
    print("Env reset complete.")
    trainer.env.step(None, None)
    print("Env step complete.")
    print("Refactor verified.")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
