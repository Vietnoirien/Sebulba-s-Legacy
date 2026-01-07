# Physics & Simulation

## 1. GPU Physics Engine ("Sim-to-Tensor")
Unlike standard frameworks that use Box2D or CPU-based physics, Sebulba's Legacy implements a custom, fully differentiable-compatible physics engine directly in PyTorch tensors.

### Key Specs
*   **Capacity**: 8,192 Concurrent Environments
*   **Precision**: Float32
*   **Map Size**: 16,000 x 9,000 units

### Parameters
*   **Pod Radius**: 400 units
*   **Friction**: 0.85
*   **Shield Duration**: 250 ms (3 simulation steps)
*   **Checkpoint Radius**: 600 units

## 2. Track Generation

### Procedural Generation
The default `generate_max_entropy` algorithm creates procedural tracks with strict guarantees:
*   **Min Distance**: 2,500 units between checkpoints.
*   **Border Buffer**: 1,800 units to prevent wall-hugging exploits.
*   **Rejection Sampling**: Ensures tracks are traversable and non-overlapping using a vectorized rejection sampling approach.

### Predefined Maps
We integrate a set of 13 Canonical Maps from the original game to ensure agents master common competitive scenarios.
*   **Usage**: Environments have a 20% chance to reset to a Predefined Map.
*   **Variety**: Includes technical tracks, speed tracks, and maps with difficult crossing points.

## 3. Reward Function
The reward function is the core driver of behavior. It uses a **Dot Product Projection** to incentivize velocity *towards the objective* rather than just raw speed.

### Dense Rewards (Per-Step)
*   **Correct Velocity**: $$R_{vel} = (\vec{v} \cdot \hat{d}) \times 0.2$$
    *   Rewards speed towards the next checkpoint.
    *   Penalizes speed moving away (backwards driving).
*   **Orientation**: Cosine similarity to the target (Weight: 1.0).
*   **Wall Collision**: Penalty proportional to impact force.

### Sparse Rewards (Events)
*   **Checkpoint Crossed**: +500.0 (Base) + Streak Multiplier.
*   **Lap Completed**: +2,000.0 * Lap Count.
*   **Win**: +5,000.0

### Team Spirit (Stage 3+)
In 2v2 modes, the "Selfish" rewards are blended with "Team" rewards using a `Team Spirit` coefficient ($\alpha$).
$$R_{total} = (1 - \alpha) R_{self} + \alpha R_{team\_avg}$$
This forces the Runner and Blocker to care about the collective team outcome, preventing the Blocker from griefing its own teammate.
