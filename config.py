
# Game Constants
WIDTH = 16000
HEIGHT = 9000
POD_RADIUS = 400.0
CHECKPOINT_RADIUS = 600.0
MAX_SPEED = 800.0
FRICTION = 0.85
MIN_IMPULSE = 120.0
SHIELD_MASS = 10.0
BOOST_THRUST = 650.0
MAX_TURN_DEGREES = 18.0
TIMEOUT_STEPS = 100 # NEVER CHANGE THIS VALUE, THIS IS A GAME CONSTRAINT

# Training / Environment Constants
MAX_LAPS = 3
MAX_CHECKPOINTS = 6
MIN_CHECKPOINTS = 3
# Curriculum Stages
STAGE_NURSERY = 0 # Learn to drive (No penalties)
STAGE_SOLO = 1    # Time Trial (Speed focus)
STAGE_DUEL = 2    # 1v1
STAGE_TEAM = 3    # 2v2
STAGE_LEAGUE = 4  # Competitive

# Graduation Thresholds
# Stage 0 (Nursery) -> 1 (Solo)
# Goal: Just hit checkpoints consistently.
STAGE_NURSERY_CONSISTENCY_THRESHOLD = 500.0

# Stage 1 (Solo) -> 2 (Duel)
# Goal: Efficiency (Speed) + Consistency
STAGE_SOLO_EFFICIENCY_THRESHOLD = 42.0
STAGE_SOLO_CONSISTENCY_THRESHOLD = 1400.0

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BotConfig:
    difficulty_noise_scale: float = 30.0
    thrust_base: float = 20.0
    thrust_scale: float = 80.0

@dataclass
class SpawnConfig:
    offsets: List[int] = field(default_factory=lambda: [500, -500, 1500, -1500])

@dataclass
class RewardScalingConfig:
    velocity_scale_const: float = 1.0 / 1000.0
    orientation_threshold: float = 0.5
    dynamic_reward_bonus: float = 1800.0

@dataclass
class TrainingConfig:
    # Hyperparameters
    lr: float = 1e-4
    gamma: float = 0.994
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    div_coef: float = 0.05 # Role Regularization Coefficient
    proficiency_penalty_const: float = 50.0
    ema_alpha: float = 0.3
    
    # Resources
    total_timesteps: int = 2_000_000_000
    num_envs: int = 16384
    num_steps: int = 256
    device: str = "cuda"
    
    # PBT Settings
    pop_size: int = 128
    evolve_interval: int = 2
    exploiter_ratio: float = 0.125
    
    # Batch Size Config (Per Agent)
    update_epochs: int = 4
    num_minibatches: int = 16
    
    @property
    def envs_per_agent(self) -> int:
        return self.num_envs // self.pop_size

    @property
    def batch_size(self) -> int:
        # Note: This is per-agent batch size (total steps collected per agent)
        # 2 * because of active pods? No, logic in ppo.py was: 2 * ENVS_PER_AGENT * NUM_STEPS
        # Assuming max 2 active pods per env.
        return 2 * self.envs_per_agent * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def split_index(self) -> int:
        num_exploiters = int(self.pop_size * self.exploiter_ratio)
        return self.pop_size - num_exploiters

@dataclass
class CurriculumConfig:
    # Stage 0 -> 1
    nursery_consistency_threshold: float = STAGE_NURSERY_CONSISTENCY_THRESHOLD
    
    # Stage 1 -> 2
    solo_efficiency_threshold: float = STAGE_SOLO_EFFICIENCY_THRESHOLD
    solo_consistency_threshold: float = STAGE_SOLO_CONSISTENCY_THRESHOLD
    
    # Stage 1 -> 2 (Duel Graduation)
    duel_consistency_wr: float = 0.82
    duel_absolute_wr: float = 0.84
    duel_consistency_checks: int = 5
    
    # Stage 2 -> 3 (Team Graduation)
    team_consistency_wr: float = 0.85
    team_absolute_wr: float = 0.88
    team_consistency_checks: int = 5

    # League Thresholds (Implicit/Monitor)
    
    # Critical Thresholds (Difficulty Adjustment)
    wr_critical: float = 0.30 # Trigger Difficulty Decrease
    wr_warning: float = 0.40 # Trigger Warning/Streak
    
    # Progression Thresholds (Difficulty Increase)
    wr_progression_standard: float = 0.70 # +0.05
    wr_progression_turbo: float = 0.80 # +0.10
    wr_progression_super_turbo: float = 0.90 # +0.20
    wr_progression_insane_turbo: float = 0.95 # +0.50
    
    # Difficulty Steps
    diff_step_decrease: float = 0.05
    diff_step_standard: float = 0.05
    diff_step_turbo: float = 0.10
    diff_step_super_turbo: float = 0.20
    diff_step_insane_turbo: float = 0.50
    
    # Nursery Specifics
    nursery_timeout_steps: int = 400

# Reward Indices
RW_WIN = 0
RW_LOSS = 1
RW_CHECKPOINT = 2
RW_CHECKPOINT_SCALE = 3
RW_VELOCITY = 4
RW_COLLISION_RUNNER = 5
RW_COLLISION_BLOCKER = 6
RW_STEP_PENALTY = 7
RW_ORIENTATION = 8
RW_WRONG_WAY = 9
RW_COLLISION_MATE = 10
RW_PROXIMITY = 11

DEFAULT_REWARD_WEIGHTS = {
    RW_WIN: 10000.0,
    RW_LOSS: 5000.0,
    RW_CHECKPOINT: 2000.0,
    RW_CHECKPOINT_SCALE: 50.0,
    RW_VELOCITY: 8.0,
    RW_COLLISION_RUNNER: 0.5,
    RW_COLLISION_BLOCKER: 1000.0,
    RW_STEP_PENALTY: 0.0,
    RW_ORIENTATION: 3.0,
    RW_WRONG_WAY: 10.0,
    RW_COLLISION_MATE: 2.0,
    RW_PROXIMITY: 5.0
}

from typing import List, Optional
# EnvConfig definition
@dataclass
class EnvConfig:
    # Mode
    mode_name: str = "standard" 
    
    # Track Generation
    track_gen_type: str = "max_entropy" # 'max_entropy', 'nursery'
    num_checkpoints_fixed: Optional[int] = None # For Nursery
    timeout_steps: int = 100 # Default global timeout
    
    # Active Pods (Simulated & Observed)
    active_pods: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
    # Bots
    use_bots: bool = False
    bot_pods: List[int] = field(default_factory=list)
    
    # Rewards
    dynamic_reward_base: float = 200.0
    step_penalty_active_pods: List[int] = field(default_factory=list)
    orientation_active_pods: List[int] = field(default_factory=list)

