# Fresh Start Implementation Plan: "Sebulba's Legacy"

## Overview
This document serves as the master blueprint for building a competitive Pod Racer AI for the "Mad Pod Racing" platform. The goal is to train a high-performance agent using PPO and Deep Reinforcement Learning, then export it to a single Python script (<100KB) using only standard libraries (`sys`, `math`).

> [!IMPORTANT]
> **Implementation Standards**:
> *   **Python**: 3.10+ (Type Hinting required)
> *   **PyTorch**: 2.0+ (MPS/CUDA support assumed)
> *   **Frontend**: React 18 + Vite + TailwindCSS
> *   **Linting**: ruff (Python), eslint (TS)

## 1. The Game: Rules & Mechanics

The target environment is **Mad Pod Racing** (CodinGame). This section defines the rules, objectives, and strategies that our agent must master.

### 1.1 Objective & Conditions
*   **Goal**: The race consists of **N laps** (Input defined, typically 3). The first team to have a pod complete all laps wins.
*   **Timeout**: A pod is eliminated if it does not pass a checkpoint within **100 turns**.
*   **Team Size**: Each team controls **2 Pods**. (Standard 1v1 match involves 4 entities on track).

### 1.2 The Map
*   **Dimensions**: 16000 x 9000 units.
*   **Checkpoints**: A sequence of circular zones (Radius 600) that must be traversed in order.
*   **Start**: All pods start at the first checkpoint, facing the second.
*   **Generation Algorithm**:
    *   **Count**: Randomly chosen between `config.min_checkpoints` and `config.max_checkpoints`.
    *   **Placement**: Random coordinates `[2000, Width-2000], [2000, Height-2000]` to avoid edge-hugging.
    *   **Constraint**: New checkpoints must be **> 3000** units away from *all* previous checkpoints to prevent overlapping clusters.
    *   **Fallback**: If placement fails (collision detected):
        *   If count > 4: Stop generating for this track (reduce count).
        *   If count <= 4: Retry until valid (ensure minimum complexity).

### 1.3 The Pod
*   **Physics**: Modeled as a frictionless disc with Radius 400.
*   **Movement**:
    *   **Rotation**: Can rotate up to **18°** per turn. (Exception: On the **First Turn**, rotation is unlimited).
    *   **Thrust**: Can apply thrust from **0 to 100**.
    *   **Inertia**: Velocity is multiplied by **0.85** (Friction) every turn.
*   **Special Actions**:
    *   **BOOST**: A one-time surge of power (Thrust 650). Available **once per GAME per TEAM** (Shared pool for the two pods).
    *   **SHIELD**: Increases mass x10 for 1 turn (Momentum conservation variant). Prevents accelerating for the next 3 turns.
    *   **First Turn Rule**: On the very first turn of the race (Turn 0), pods can rotate freely (no 18° limit).

### 1.4 Strategies (The Meta)
*   **The Runner**: One pod acts as the primary racer. Its goal is to optimize trajectory, manage boost, and finish laps as fast as possible.
*   **The Blocker**: The second pod acts as a defender. Its goal is to intercept the opponent's Runner, colliding with them to disrupt their trajectory or push them off-course.

#### The "Dynamic Leadership" Algorithm
Instead of forcing the AI to *choose* a role, we **assign** it based on race state. We feed a boolean `is_leader` into the network context.

1.  **Score Calculation**: `Score = (Laps * 100) + CheckpointID`.
2.  **Comparison**: The pod with the highest Score is the **Runner**.
3.  **Tie-Breaker**: If Scores are equal, the pod closer to the next checkpoint is the **Runner**.
4.  **Hysteresis (Anti-Flicker)**: To swap roles, the trailing pod must overtake the leader by at least **1 full checkpoint** (or significant distance) to prevent chaotic role switching every frame.
    *   *Effect*: The AI knows "I am the Runner" and activates the Runner-specific Policy Head/Reward weights.

## 2. Core Architecture: The "DeepSets" Model

We will use a **Permutation Invariant (DeepSets)** architecture. This allows the model to handle "Team" and "Opponents" as sets of entities rather than fixed input slots, making it robust to player index swaps and reducing parameter count.

### 2.1 Model Design
The model consists of three stages: **Encoder**, **Aggregator**, and **Policy Head**.

#### A. The Feature Encoder (Shared MLP)
Processes a single entity (e.g., an opponent pod) relative to the controlled pod.
*   **Normalization Constants**:
    *   `SCALE_POS = 1.0 / 16000.0` (Map Width)
    *   `SCALE_VEL = 1.0 / 1000.0` (Approx Max Speed 800 + Margin)
    *   `SCALE_ANGLE = 1.0` (Intrinsic Cos/Sin)
*   **Input (13 dim)**:
    *   Relative Position $(dx, dy) \times$ `SCALE_POS`
    *   Relative Velocity $(dvx, dvy) \times$ `SCALE_VEL`
    *   Relative Angle (Cos, Sin) of entity
    *   Distance to entity $\times$ `SCALE_POS`
    *   Entity's Next Checkpoint Info (RelPos $\times$ `SCALE_POS`, Dist $\times$ `SCALE_POS`, Angle Cos/Sin)
    *   Entity Properties (ShieldCooldown / 3.0, IsBoostAvailable)
        *   *Note*: For opponents, these must be inferred from velocity changes/mass impacts or set to 0 if untrackable.
*   **Layers**: `Linear(13, 32) -> ReLU -> Linear(32, 16)`
*   **Output**: 16-dimensional "Latent Entity Embedding".

#### B. The Aggregator (Symmetric Function)
Combines embeddings from multiple entities into a fixed-size representation.
*   **Logic**: `Global_Context = MaxPool(Encoder(Opp1), Encoder(Opp2), Encoder(Opp3))`
*   **Why**: `Max` operation is permutation invariant. Whether input is `[A, B]` or `[B, A]`, the result is identical.

#### C. The Policy Head (Decision Maker)
Decides actions for *Self*.
*   **Input**: `[Self_Features(14), Global_Context(16), Next_Checkpoint_Features(6)]`
    *   *Self_Features* includes absolute velocity, angle, boost state, shield state, etc.
*   **Layers**: `Linear(36, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, OutputDim)`
*   **Output (Action Space)**:
    *   **Thrust (Continuous)**: `Sigmoid` [0, 1] (mapped to 0-100)
    *   **Angle (Continuous)**: `Tanh` [-1, 1] (mapped to -18° to +18°)
    *   **Shield (Discrete)**: Probability > Threshold triggers SHIELD.
    *   **Boost (Discrete)**: Probability > Threshold triggers BOOST.

## 3. The Simulation Engine (Dual-Stack)

To leverage the **RTX 5070**, we will implement a logic-preserving Dual-Stack simulation architecture.

### 3.1 Physics Specification (The Truth)
All implementations (CPU & GPU) must strictly adhere to these constants and formulas.

#### Constants
```python
WIDTH = 16000
HEIGHT = 9000
POD_RADIUS = 400.0
CHECKPOINT_RADIUS = 600.0
MAX_SPEED = 800.0     # Cap not always enforced hard, but good for norm
FRICTION = 0.85
MIN_IMPULSE = 120.0
SHIELD_MASS = 10.0    # Normal mass = 1.0
BOOST_THRUST = 650.0
MAX_TURN_DEGREES = 18.0
```

#### Collision Formula (Elastic)
Collisions happen when `dist(P1, P2) < 2 * POD_RADIUS` (800).
1.  **Normal Vector**: $N = P_2 - P_1$, normalized.
2.  **Relative Velocity**: $V_{rel} = V_1 - V_2$.
3.  **Impact Force**:
    $$ F = \frac{N \cdot V_{rel}}{\frac{1}{M_1} + \frac{1}{M_2}} $$
4.  **Min Impulse**: If $F < 120$, set $F = 120$ (The "Bumper Car" rule).
5.  **Impulse Vector**: $J = -N \times F$.
6.  **New Velocities**:
    *   $V_1' = V_1 + J / M_1$
    *   $V_2' = V_2 - J / M_2$
7.  **Separation**: Because simulation is discrete, pods might overlap. Move them apart along normal $N$ by `(Radius*2 - Dist) / 2` each.

#### Movement Step Order
1.  Rotate to desired angle (clamped $\pm 18^\circ$ relative to current).
2.  Add Thrust to Velocity.
3.  Move Position (`pos += vel`).
4.  Resolve Collisions (as above).
5.  Apply Friction (`vel *= 0.85`), then **Truncate** (towards zero, `int(vel)`).
6.  Round Position to nearest integer (`round(pos)`).

### 3.2 GPU Engine (Training)
*   **Implementation**: PyTorch Tensors (CUDA).
*   **Performance**: Massive batching (e.g., 4096 concurrent races).
*   **Usage**: Primary environment for PPO training loop.
*   **Physics**: Vectorized implementation of the rules above.

#### Vectorized Collision Solver (The "Scatter-Add" Algorithm)
To handle multi-body collisions (e.g., A hits B, B hits C) without Python loops, we use a **Fixed-Step Iterative Impulse Solver** with scatter-reduction.

1.  **State**: `Pos` and `Vel` tensors of shape `[Batch, 4, 2]`.
2.  **Topology**: We statically define the 6 unique pairs for 4 pods: `[(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]`.
3.  **Solver Loop**: Run `K=4` iterations per physics step.
    *   **Gather**: Extract `P1, P2, V1, V2` for all 6 pairs -> Shape `[Batch, 6, 2]`.
    *   **Detect**: Compute distances. Mask `collisions = dist < 2*RADIUS`.
    *   **Solve**: Compute `Impulse Vector` and `Separation Vector` (Position Correction) for all pairs simultaneously.
    *   **Accumulate**: Create zero-tensors `dVel` and `dPos`. Use `index_add_` (Scatter Add) to sum up vectors for pods involved in multiple collisions (e.g., Pod 0 hitting both 1 and 2).
    *   **Apply**: `Vel += dVel`, `Pos += dPos`.
    *   *Repeat*: Subsequent iterations allow the energy to propagate (A->B->C).

### 3.3 CPU Engine (Validation & Export)
*   **Implementation**: Pure Python (`math` lib only).
*   **Usage**: Golden Standard validation and final export code.
*   **Consistency Check**: Periodically compare `GPU(State)` vs `CPU(State)`.

## 4. Training Pipeline

### 4.1 Algorithm: PPO (Proximal Policy Optimization)
*   **Reference**: Follow **CleanRL** implementation standards.
*   **Hardware Target**: RTX 5070 (High Memory/Compute).
*   **Configuration**:
    *   **Total Timesteps**: `2,000,000,000` (2 Billion). Given the GPU physics, we expect 50k-100k SPS (Steps Per Second), meaning ~6-12 hours for convergence.
    *   **Num Envs (Parallel)**: `4096`. Massive parallelism to stabilize the advantage estimator.
    *   **Num Steps (Rollout)**: `128`.
    *   **Total Batch Size**: `524,288` (4096 * 128).
    *   **Mini-Batch Size**: `32,768` (Total / 16). Larger batches = better gradient estimates.
    *   **Update Epochs**: `4`. Keep low to preserve sample efficiency with such large batches.
    *   **Learning Rate**: `3e-4` (Linear Decay to 0).
    *   **Gamma (Discount)**: `0.994`. Long horizon focus (laps are long).
    *   **GAE Lambda**: `0.95`.
    *   **Clip Range**: `0.2`.
    *   **Ent Coef**: `0.01`.
    *   **Max Grad Norm**: `0.5`.
### 4.2 League-Based Self-Play
*   **Architecture**: File-System based Registry.
*   **Storage Path**: `data/checkpoints/`
*   **Manifest**: `data/league.json` (Tracks ELO, Creation Time, Parent ID).
*   **ELO System**:
    *   **Formula**: Standard Rating $R_a' = R_a + K(S_a - E_a)$.
    *   **K-Factor**: `32` (Standard for games).
    *   **Starting ELO**: `1000`.
*   **Pruning Strategy (The "Hall of Fame")**:
    *   **Capacity**: Max `100` models.
    *   **Protection (Anchors)**: Top `10` Highest ELO models are *never* deleted ("Gatekeepers").
    *   **Selection**: When full, remove the *oldest* model that is *not* an Anchor.
*   **Matchmaking**: 
    - 80% vs **Latest Self** (Gradient optimization).
    - 20% vs **League Opponent** (Uniform sampling from the "Hall of Fame" to ensure robustness against forgotten strategies).

### 4.3 Curriculum Strategy (The "Sebulba" Method)
Inspired by successful RL projects (OpenAI Five, GT Sophy), we introduce complexity gradually. We use **Automatic Curriculum Learning (ACL)** where stage transitions are triggered by performance metrics (Win Rate, Lap Time).

#### Stage 1: Solo Pilot (Physics & Racing Lines)
*   **Scenario**: 1 Pod (Runner) vs Empty Track.
*   **Goal**: Master movement, friction, and checkpoint validation.
*   **Opponent**: None.
*   **Reward**:
    *   Dense: Velocity to Ckpt (+), Distance to Ckpt (-).
    *   Penalty: Timeout (-), Reverse usage (-).
*   **Exit Condition**: Complete 3 laps consistently < 150 turns.

#### Stage 2: The Duelist (Collision & Blocking)
*   **Scenario**: 1v1 (Runner vs Simple Bot).
*   **Goal**: Learn to overtake and bump.
*   **Opponent**: A hardcoded "path-follower" bot that sometimes makes mistakes.
*   **Reward**:
    *   Transition to hybrid: 50% Dense (Speed), 50% Sparse (Win/Loss).
    *   Collision Reward: Positive if vector points to opponent, Negative if being hit from behind.
*   **Exit Condition**: Win Rate > 85% vs Simple Bot.

#### Stage 3: The Tactician (2v2 Coordination)
*   **Scenario**: Full 2v2 Game.
*   **Goal**: Optimize dual-control (Runner + Blocker).
*   **Opponent**: Self-Play (Current Policy).
*   **Reward**:
    *   **Team Spirit**: Shared reward pool. If *any* pod wins, both get +1.
    *   **Role Specialization**:
        *   Runner: Heavily penalized for collisions (should avoid).
        *   Blocker: Highly rewarded for collisions with *Enemy Runner*.
*   **Exit Condition**: Win Rate stabilizes (Nash Equilibrium approach).

#### Stage 4: The League (Robustness)
*   **Scenario**: 2v2 vs Historical Pool.
*   **Goal**: Generalization and exploit prevention.
*   **Opponent**: Randomly sampled past versions.
*   **Reward**: 100% Sparse (Win/Loss).

### 4.4 Reward Engineering (The "Team Spirit")

We adopt a **Potential-Based Reward Shaping** approach (Ng et al., 1999) to guarantee policy invariance, combined with a **Team Spirit Blending** factor (OpenAI Five).

#### The Master Formula
$$ R_{total} = (1 - \tau) \cdot R_{dense} + \tau \cdot R_{sparse} $$
$$ R_{final} = (1 - \beta) \cdot R_{self} + \beta \cdot R_{teammate} $$

*   **$\tau$ (Curriculum Factor)**: Starts at `0.0` (Pure Guidance) -> Anneals to `1.0` (Pure Objective).
*   **$\beta$ (Spirit Factor)**: Typically `0.3` to `0.5`. Ensures agents don't sacrifice the game for personal stats.

#### Component 1: Sparse Rewards (The Objective)
*   **Win**: `+1000` (First team to cross finish line).
*   **Loss**: `-1000` (Enemy wins).
*   **Timeout**: `-500` (Eliminated).

#### Component 2: Dense Rewards (The Guidance)
Calculated every step `t`.

**A. Progress (Potential Field)**
$$ R_{velocity} = (\Phi_{t+1} - \Phi_{t}) $$
Where $\Phi(s)$ is the "Progress" score (Distance covered along the center-line of the track).
*   *Why?* Proven to prevent "looping" behavior. If the agent goes back, it loses reward. If it goes forward, it gains it.

**B. Checkpoint Gate**
*   **Pass Checkpoint**: `+100 + (1000 / StepsTakenSinceLast)`
    *   *Incentive*: Pass checkpoints FAST.

**C. Collision (Role Specific)**
Let $I$ be the impact impulse magnitude.
*   **Runner**:
    *   If Out of Bounds: `-10.0` (Encourage staying within map).
    *   If hitting Enemy: `-0.5 * |I|` (Avoid disruption).
*   **Blocker**:
    *   If hitting Enemy Runner: `+2.0 * |I|` (Encourage hard hits).
    *   If pushing Enemy Backwards: `+5.0 * (EnemyVelocity_Old - EnemyVelocity_New)`.

**D. Energy Efficiency (Regularization)**
*   **Thrust Penalty**: `-0.01 * Thrust` (Prefer gliding when speed is sufficient).
*   **Shield Penalty**: `-10.0` (Use sparingly).


## 5. The Interface: Human Supervision & Control

A powerful AI needs a powerful cockpit. The interface is not just a viewer; it is the control center for the entire operation.

### 5.1 The Dashboard (Control Room)
*   **Live Telemetry**: Real-time graphs for Reward Mean, Entropy, Value Loss, and Win Rate.
*   **Health Monitor**: Systems status (GPU Memory, FPS, TPS).
*   **Action Log**: A scrolling log of major events (Checkpoints reached, Laps finished, New Best Model).

### 5.2 Interactive Configuration (Hot-Reloading)
*   **Architecture**: Thread-Safe Singleton `ConfigManager`.
*   **Data Flow**:
    1.  **Read**: `PPO_Trainer` calls `ConfigManager.get_snapshot()` **once** at the start of each iteration. This ensures parameters (LR, Weights) remain constant during the gradient step.
    2.  **Write**: `FastAPI` (UI) calls `ConfigManager.update(new_params)` protected by a `threading.Lock`.
*   **Hyperparameter Tuning**: Sliders adjust `learning_rate` and `ent_coef` instantly.
*   **Reward Shaping**: Dynamic weights for distinct rewards (e.g., "Increase 'Checkpoint' reward").
*   **Curriculum Control**: Manually advance or regress the curriculum stage.

### 5.3 Manual Intervention

*   **Scenario Builder**: (Future) Place pods and checkpoints to test specific edge cases.
*   **Checkpoint Management**: One-click Save, Load, and Delete for model checkpoints.

### 5.4 Visualization (The "Eye")
*   **Race Canvas**: 60fps rendering with interpolation.
*   **Debug Overlays**:
    *   **Vectors**: Velocity, Thrust, Impact Normals.
    *   **Lidar**: Visualization of the agent's sensory input (e.g., rays to obstacles).

*   **Live Training View**: Watch the AI learn in real-time.
*   **Physics Debugger**: Overlay showing velocity vectors, collision points, and predicted trajectories.


## 6. The Export System (The "Transpiler")

How to fit a neural network into a `sys+math` script?

### 6.1 Weight Compression
1.  **Quantization**: Convert all `float32` weights to `int8` (mapped -127 to 127).
2.  **Encoding**: Encode bytes to **Base85** (more efficient than Ascii85 for code).
3.  **Constraint**: Final script must be **< 100,000 characters**. (Strict Hard Limit).

### 6.2 The "Micro-Inference" Library
We include a compact class in the final script. **CRITICAL**: This library must be Pure Python (No `numpy`).
We must implement these functions from scratch using `math`:
*   `dot_product(vec_a, vec_b)`: For matrix multiplication.
*   `add(vec_a, vec_b)`: For bias addition.
*   `tanh(x)`: `(math.exp(2*x) - 1) / (math.exp(2*x) + 1)`
*   `sigmoid(x)`: `1 / (1 + math.exp(-x))`
*   `relu(x)`: `max(0, x)`

```python
class NN:
    def __init__(self, data_str, scale):
        self.weights = self.decode(data_str, scale)
    
    def decode(self, blob, display_scale):
        # Pure Python Base85/Ascii85 Decoder (No Imports)
        w = []
        val = 0
        count = 0
        for char in blob:
            # Map '!' (33) to 0, ... 'u' (117) to 84
            val = val * 85 + (ord(char) - 33)
            count += 1
            if count == 5:
                # Unpack 4 bytes (Big Endian)
                for i in range(4):
                    # Extract byte, map 0..255 to -128..127
                    b = (val >> (24 - i*8)) & 0xFF
                    w.append((b - 256 if b > 127 else b) * display_scale)
                val = 0
                count = 0
        return w

    def matmul(self, input_vec, weight_idx, rows, cols):
        # Slice weights from flat array
        w_slice = self.weights[weight_idx : weight_idx + rows * cols]
        result = []
        for r in range(rows):
            acc = 0.0
            for c in range(cols):
                acc += input_vec[c] * w_slice[r * cols + c]
            result.append(acc)
        return result
```

### 6.3 The Game Loop (I/O Boilerplate)
The final exported script must wrap the `NN` library with the specific CodinGame I/O protocol. This layer handles logical adaptation (e.g., converting 0-1 model outputs to game coordinates).

```python
# [Insert Micro-Inference Class Here]

def solve():
    # 1. Initialization
    try:
        laps = int(input())
        checkpoint_count = int(input())
        checkpoints = []
        for _ in range(checkpoint_count):
            checkpoints.append(list(map(int, input().split())))
            
        # Initialize Model (Weights injected dynamically during export)
        # model = NN(WEIGHTS_DATA, SCALES)
        
        # --- State Tracking ---
        # Game inputs are instantaneous, so we must track hidden state.
        turn = 0
        team_boost_available = True
        my_shield_cooldowns = [0, 0] # [Pod0, Pod1]

        while True:
            # 2. Input Parsing
            pods = []
            # My Pods
            for i in range(2):
                x, y, vx, vy, angle, next_cp = map(int, input().split())
                pods.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'angle': angle, 'next_cp': next_cp, 'owner': 'me', 'id': i})
            # Opponent Pods
            for i in range(2):
                x, y, vx, vy, angle, next_cp = map(int, input().split())
                pods.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'angle': angle, 'next_cp': next_cp, 'owner': 'opp', 'id': i+2})

            # 3. Model Inference
            # features = extract_features(pods, checkpoints, turn, team_boost_available, my_shield_cooldowns)
            # actions = model.forward(features) # Returns [AngleOffset, Thrust, ShieldProb, BoostProb]

            # 4. Output Formatting
            for i in range(2):
                # Unpack Action
                # angle_offset_deg = actions[i][0] * 18.0
                # thrust_val = int(actions[i][1] * 100)
                # shield_trigger = actions[i][2] > 0.5
                # boost_trigger = actions[i][3] > 0.5
                
                # A. Handle Constraints
                current_angle = pods[i]['angle']
                
                # Logic: Convert Angle Offset to Target Point
                # We want to point to: CurrentAngle + Offset
                # TargetX = X + cos(a) * 1000, TargetY = Y + sin(a) * 1000
                import math
                target_angle_rad = math.radians(current_angle + angle_offset_deg)
                target_x = int(pods[i]['x'] + math.cos(target_angle_rad) * 10000)
                target_y = int(pods[i]['y'] + math.sin(target_angle_rad) * 10000)
                
                # B. Handle Boost & Shield Logic
                power = str(thrust_val)
                if shield_trigger and my_shield_cooldowns[i] == 0:
                    power = "SHIELD"
                    my_shield_cooldowns[i] = 3 + 1 # +1 because it decrements this turn
                elif boost_trigger and team_boost_available:
                    power = "BOOST"
                    team_boost_available = False
                
                # Decrement Cooldowns
                if my_shield_cooldowns[i] > 0:
                    my_shield_cooldowns[i] -= 1

                print(f"{target_x} {target_y} {power}")
            
            turn += 1
                
    except EOFError:
        pass

if __name__ == "__main__":
    solve()
```

## 7. Implementation Stages

### Phase 1: Foundation
*   [x] Set up directory structure.
*   [x] Implement `simulation/cpu_physics.py`: The Reference.
*   [x] Implement `simulation/gpu_physics.py`: The Accelerator.
*   [x] Create `tests/test_physics_parity.py`: Verify GPU matches CPU.

### Phase 2: The Interface (GUI & Protocol)

#### 2.1 WebSocket Protocol Schema (`/ws/telemetry`)
*   **Format**: JSON text frames.
*   **Direction**: **Server -> Client (Telemetry)**
    *   Frequency: 60Hz (broadcast).
    *   Schema:
        ```json
        {
          "type": "telemetry",
          "step": 12345,
          "match_id": "uuid-v4",
          "stats": {
             "fps_physics": 45000.0,
             "fps_training": 128.0,
             "reward_mean": 12.5,
             "win_rate": 0.45,
             "active_model": "checkpoint_v100"
          },
          "race_state": {
             # Only Env 0 (Visualization)
             "pods": [
                {"id": 0, "team": 0, "x": 1000, "y": 2000, "angle": 0.5, "boost": 1, "shield": 0},
                {"id": 1, "team": 0, "x": 1200, "y": 2100, "angle": 0.1, "boost": 1, "shield": 0},
                {"id": 2, "team": 1, "x": 5000, "y": 8000, "angle": -0.5, "boost": 0, "shield": 10},
                {"id": 3, "team": 1, "x": 5200, "y": 8100, "angle": -0.1, "boost": 1, "shield": 0}
             ],
             "checkpoints": [{"x": 5000, "y": 5000}, ...]
          }
        }
        ```

*   **Direction**: **Client -> Server (Control)**
    *   Schema (Command):
        ```json
        {
          "type": "command", 
          "action": "START" | "STOP" | "RESET" | "SAVE_CHECKPOINT"
        }
        ```
    *   Schema (Config Update):
        ```json
        {
          "type": "config",
          "payload": {
             "learning_rate": 0.0003,
             "reward_weights": {"velocity": 1.0, "collision": 0.5}
          }
        }
        ```

#### 2.2 UI/UX Components
*   **Control Room Layout (Single Page)**:
    *   **Header**: Connection Status (Green/Red), FPS Counter, Current Model Name.
    *   **Left Sidebar (The Mechanic)**:
        *   `ConfigPanel`: Accordion lists for `Hyperparameters` and `Reward Shaping` (Sliders/Inputs).
        *   `ControlPanel`: Start/Stop/Reset and Save Checkpoint buttons.
    *   **Center Stage (The Track)**:
        *   `RaceCanvas`: React-Konva or Canvas2D.
        *   Renders the 16000x9000 map scaled to fit.
        *   Draws trails for recent positions (Visualization of trajectories).
    *   **Right Sidebar (The Analyst)**:
        *   `StatsPanel`: Sparklines for Loss/Reward history.
        *   `LeagueTable`: Top 5 Past Agents by ELO.
        *   `LogStream`: Scrolling text of server events (e.g., "New Best Model Saved").


### Phase 3: The Learner
*   [x] **Implement `models/deepsets.py`**:
    *   Construct the Permutation Invariant Network (DeepSets) to handle variable inputs (1 Teammate, 2 Opponents).
    *   Implement `FeatureEncoder`: Shared MLP for encoding entity states (Teammate, Opponents).
    *   Implement `Aggregator`: Symmetric function (Max/Mean pooling) to create a global context from opponent embeddings.
    *   Implement `PolicyHead`: Multi-headed output for Thrust (0-100), Angle (-18 to +18), Shield, and Boost.
*   [ ] **Implement Game Mechanics & Strategy**:
    *   **2 vs 2 Setup**: Configure environment to support 2 teams of 2 pods each.
    *   **Role Definitions**: Define logic for "Runner" (Focus on Checkpoints/Speed) and "Blocker" (Focus on disrupting Enemy Runner).
    *   **Reward Shaping**:
        *   *Runner*: Dense rewards for velocity towards next checkpoint, passing checkpoints.
        *   *Blocker*: Rewards for proximity to enemy runner, collisions with enemies, and velocity relative to enemy runner.
*   [x] **Implement `training/ppo.py`**:
    *   high-performance GPU-based PPO training loop.
    *   Support for "Team Spirit" rewards (combining individual and team success).
*   [x] **Implement `training/self_play.py`**:
    *   League Manager to store and sample past policies.
    *   Queue system for matchmaking (80% latest self, 20% past versions).

### Phase 4: The Integrator
*   [x] Implement `export.py`: Quantization and string generation.
*   [x] Verify exported script logic.


## 8. Directory Structure
```
/
├── simulation/
│   ├── cpu_physics.py   # Ref implementation
│   ├── gpu_physics.py   # Training env
│   └── env.py           # Gym Wrapper
├── web/                 # GUI Components
│   ├── backend/         # FastAPI
│   └── frontend/        # React
├── training/
│   ├── ppo.py           # Trainer
│   └── self_play.py     # League Manager
├── models/
│   └── deepsets.py      # Architecture
├── export/
│   └── quantization.py
└── main.py
```
.venv                    # Virtual environment with dependencies