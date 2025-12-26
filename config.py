
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

# Training / Environment Constants
MAX_LAPS = 3
TIMEOUT_STEPS = 100
MAX_CHECKPOINTS = 6
MIN_CHECKPOINTS = 3
# Curriculum Stages
STAGE_SOLO = 0
STAGE_DUEL = 1
STAGE_TEAM = 2
STAGE_LEAGUE = 3

# Graduation Thresholds (Stage 0 -> 1)
# Efficiency: Avg Steps per Checkpoint (Lower is better). 30 = ~Decent driving.
STAGE_SOLO_EFFICIENCY_THRESHOLD = 28.0
# Consistency: Sum of CPs passed by all 128 envs in a generation (512 steps).
# Max theoretical ~3072 (128 * 24). Goal 3000 = ~80% success rate.
STAGE_SOLO_CONSISTENCY_THRESHOLD = 2400.0
