
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
STAGE_NURSERY_CONSISTENCY_THRESHOLD = 2000.0

# Stage 1 (Solo) -> 2 (Duel)
# Goal: Efficiency (Speed) + Consistency
STAGE_SOLO_EFFICIENCY_THRESHOLD = 30.0
STAGE_SOLO_CONSISTENCY_THRESHOLD = 2000.0
