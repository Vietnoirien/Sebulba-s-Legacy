
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
TIMEOUT_PENALTY_STANDARD = 3500.0
TIMEOUT_PENALTY_DUEL = 3000.0

# Training / Environment Constants
MAX_LAPS = 3
MAX_CHECKPOINTS = 6
MIN_CHECKPOINTS = 3
# Curriculum Stages
STAGE_NURSERY = 0 # Learn to drive (No penalties)
STAGE_SOLO = 1    # Time Trial (Speed focus)
STAGE_DUEL = 2    # 1v1
STAGE_INTERCEPT = 3 # Blocker Academy (PvE Blocking)
STAGE_TEAM = 4    # 2v2
STAGE_LEAGUE = 5  # Competitive

# Graduation Thresholds
# Stage 0 (Nursery) -> 1 (Solo)
# Goal: Just hit checkpoints consistently.
STAGE_NURSERY_CONSISTENCY_THRESHOLD = 500.0

# Stage 1 (Solo) -> 2 (Duel)
# Goal: Efficiency (Speed) + Consistency
STAGE_SOLO_EFFICIENCY_THRESHOLD = 40.0
STAGE_SOLO_CONSISTENCY_THRESHOLD = 3000.0
STAGE_SOLO_PENALTY_CONSISTENCY_THRESHOLD = 1000.0
STAGE_SOLO_PENALTY_EFFICIENCY_THRESHOLD = 55.0
STAGE_SOLO_PENALTY_EXIT_EFFICIENCY_THRESHOLD = 65.0
STAGE_SOLO_DYNAMIC_STEP_PENALTY = 20.0

from dataclasses import dataclass, field
from typing import List, Optional, Dict

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
    div_coef: float = 0.005 # Role Regularization Coefficient (Reduced from 0.05 to prevent apathy)
    proficiency_penalty_const: float = 50.0
    ema_alpha: float = 0.3
    
    # Resources
    total_timesteps: int = 2_000_000_000
    num_envs: int = 8192
    num_steps: int = 512
    device: str = "cuda"
    
    # PBT Settings
    pop_size: int = 64
    evolve_interval: int = 2
    exploiter_ratio: float = 0.125
    max_checkpoints_to_keep: int = 5
    
    # Batch Size Config (Per Agent)
    update_epochs: int = 4
    num_minibatches: int = 64
    
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
    solo_penalty_consistency_threshold: float = STAGE_SOLO_PENALTY_CONSISTENCY_THRESHOLD
    solo_penalty_efficiency_threshold: float = STAGE_SOLO_PENALTY_EFFICIENCY_THRESHOLD
    solo_penalty_exit_efficiency_threshold: float = STAGE_SOLO_PENALTY_EXIT_EFFICIENCY_THRESHOLD
    solo_dynamic_step_penalty: float = STAGE_SOLO_DYNAMIC_STEP_PENALTY
    solo_min_win_rate: float = 0.90
    
    # Stage 2 (Duel) -> 3 (Team) Graduation
    # Graduation triggers when Bot Difficulty >= graduation_difficulty
    # AND Win Rate >= graduation_win_rate for graduation_checks
    duel_graduation_difficulty: float = 0.80
    duel_graduation_win_rate: float = 0.65
    duel_graduation_checks: int = 5
    
    # Stage 3 (Team) -> 4 (League) Graduation
    team_graduation_difficulty: float = 0.85
    team_graduation_win_rate: float = 0.70
    team_graduation_checks: int = 5
    
    # Team Stage Start Difficulty
    team_start_difficulty: float = 0.6
 
    # Critical Thresholds (Difficulty Adjustment)
    wr_critical: float = 0.25 # Trigger Difficulty Decrease
    wr_warning: float = 0.45 # Trigger Warning/Streak
    
    # Progression Thresholds (Difficulty Increase)
    wr_progression_standard: float = 0.55 # +0.01
    wr_progression_turbo: float = 0.60 # +0.02
    wr_progression_super_turbo: float = 0.70 # +0.05
    wr_progression_insane_turbo: float = 0.85 # +0.1
    
    # Difficulty Steps
    diff_step_decrease: float = 0.02
    diff_step_standard: float = 0.01
    diff_step_turbo: float = 0.02
    diff_step_super_turbo: float = 0.05
    diff_step_insane_turbo: float = 0.10
    
    # Nursery Specifics
    nursery_timeout_steps: int = 300

# Reward Indices
# Reward Indices
RW_WIN = 0
RW_LOSS = 1
RW_CHECKPOINT = 2
RW_CHECKPOINT_SCALE = 3
RW_PROGRESS = 4 # Replaces RW_VELOCITY (Index 4 recycled to keep tensor size consistent if loading old models, though meaning changes)
RW_COLLISION_RUNNER = 5
RW_COLLISION_BLOCKER = 6
RW_STEP_PENALTY = 7
RW_ORIENTATION = 8
RW_WRONG_WAY = 9
RW_COLLISION_MATE = 10
RW_PROXIMITY = 11
RW_MAGNET = 12 # Proximity Pull
RW_RANK = 13 # Rank Improvement
RW_LAP = 14 # Lap Completion
RW_DENIAL = 15 # Deny Enemy Progress (Blocker Only)
LAP_REWARD_MULTIPLIER = 1.5

DEFAULT_REWARD_WEIGHTS = {
    RW_WIN: 10000.0,
    RW_LOSS: 2000.0,
    RW_CHECKPOINT: 500.0, # Reduced from 2000.0 to 500.0 (User Request)
    RW_CHECKPOINT_SCALE: 50.0,
    RW_PROGRESS: 0.2, # Scaled down to prevent overpowering Checkpoint (2000) 
    RW_COLLISION_RUNNER: 0.5,
    RW_COLLISION_BLOCKER: 5.0, # Reduced from 1000.0 (Impact is ~800, so 5*800=4000 ~ 2 Checkpoints)
    RW_STEP_PENALTY: 10.0, # Increased to 10.0 to make Time Cost significant (Speed Incentive)
    RW_ORIENTATION: 1.0, # Reduced to soft guidance
    RW_WRONG_WAY: 10.0,
    RW_COLLISION_MATE: 5.0, # Increased from 2.0 (User Request) to discourage friendly fire
    RW_PROXIMITY: 5.0,
    RW_MAGNET: 10.0, # Proximity Pull
    RW_RANK: 500.0, # Rank Change
    RW_LAP: 2000.0, # RW_LAP (New) - Index 14 manually assigned
    RW_DENIAL: 0.5 # Deny Reward (New) - Balances with Collision (0.5 * 600u = 300/step => ~3000/sec)
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
    # dynamic_reward_base Removed (Legacy)
    step_penalty_active_pods: List[int] = field(default_factory=list)
    orientation_active_pods: List[int] = field(default_factory=lambda: [0])
    # Optional map for Fixed Roles (PodID -> RoleID). e.g. {0: 0, 1: 1} to force Pod0=Blocker, Pod1=Runner
    fixed_roles: Optional[Dict[int, int]] = None
