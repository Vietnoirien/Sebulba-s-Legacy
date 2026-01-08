# System Documentation: Observations & Rewards

> [!NOTE]
> This document details the currently implemented Observation Space and Reward Function as of **Tier 2 ("The Wrestler")** update.

## 1. Observation Space
The model consumes a hierarchical observation set processed by the **DeepSets** architecture.

### A. Self Features (`self_obs`)
**Dimension: 15**
Encodes the physical and strategic state of the ego-agent.
| Index | Feature | Description |
| :--- | :--- | :--- |
| 0-1 | `v_local` | Velocity vector rotated to body frame (Forward, Right). |
| 2-3 | `t_vec_l` | Vector to Next Checkpoint (Body Frame). |
| 4 | `dest` | Distance to Next Checkpoint (Normalized). |
| 5-6 | `align` | Cos/Sin of angle to Next Checkpoint. |
| 7 | `shield` | Shield Cooldown (Normalized). |
| 8 | `boost` | Boost Available (1.0 or 0.0). |
| 9 | `timeout` | Turns until timeout (0-100 Normalized). |
| 10 | `lap` | Current Lap (Normalized). |
| 11 | `leader` | Is Leader/Runner flag (1.0 if Runner, 0.0 if Blocker). |
| 12 | `v_mag` | Speed magnitude scalar. |
| 13 | `pad` | Padding (Reserved). |
| 14 | `rank` | Race Rank (Normalized). |

### B. Entity Features (`entity_obs`)
**Dimension: 14**
Perceived state of other pods (Teammates & Enemies). Processed via Permutation Invariant DeepSet.
| Index | Feature | Description |
| :--- | :--- | :--- |
| 0-1 | `dp_local` | Relative Position vector (Body Frame). |
| 2-3 | `dv_local` | Relative Velocity vector (Body Frame). |
| 4 | `rel_cos` | Cosine of relative angle. |
| 5 | `rel_sin` | Sine of relative angle. |
| 6 | `dist` | Distance to entity. |
| 7 | `is_mate` | Team affinity (1.0 = Teammate, 0.0 = Enemy). |
| 8 | `o_shield` | Entity Shield Status (Active/Cooldown). |
| 9-10 | `ot_vec_l` | Entity's vector to *their* Next Checkpoint. |
| 11 | `o_is_run` | Is Runner flag. |
| 12 | `o_rank` | Entity Rank. |
| 13 | **`o_timeout`** | **[NEW]** Entity Timeout urgency (Allows exploiting desperate enemies). |

### C. Checkpoint Features (`cp_obs`)
**Dimension: 10**
Foresight of the track layout.
| Index | Feature | Description |
| :--- | :--- | :--- |
| 0-1 | `t_vec_l` | Vector to CP 1 (Next). |
| 2-3 | **`v12_l`** | **[NEW]** Vector CP 1 -> CP 2. Allows setting up for the next turn. |
| 4-5 | **`v23_l`** | **[NEW]** Vector CP 2 -> CP 3. Extended lookahead. |
| 6 | `cps_left` | Normalized Checkpoints remaining. |
| 7 | **`corner`** | **[NEW]** Cosine of the angle between `Me->CP1` and `CP1->CP2`. (Sharpness). |
| 8 | `max_spd` | Heuristic Max Speed Approach for corner. |
| 9 | `pad` | Padding. |

---

## 2. Reward Function
Rewards are composed of **Dense** (per-step) and **Sparse** (event-based) signals.

> [!IMPORTANT]
> **Reward Annealing**: All Dense rewards are annealed using the `tau` schedule defined in the curriculum. 
> As the agent graduates through stages, `tau` increases, reducing the strength of dense signals and forcing the agent to rely on Sparse (Win/Loss) rewards. This prevents over-fitting to specific dense "hacks" in later stages.

### Core Rewards
*   **Progress**: Velocity projected towards next checkpoint.
*   **Checkpoint**: Bonus (+500) for passing CP.
*   **Win/Loss**: Large sparse reward (+10k/-2k).

### Tier 2 Interaction Rewards ("The Wrestler")
Specific logic for **Blocker** agents (Pod 1/3) to create effective interference.

#### A. Zone Control (Intercept)
Rewards the blocker for positioning themselves in the **future path** of the enemy.
*   **Logic**: Projects `Vector(Me -> Enemy)` onto `Vector(Enemy -> Target)`.
*   **Effect**: If positive, I am "downstream" of the enemy -> Reward. Encourages "Parking" and "Screening" rather than just chasing.

#### B. Timeout Pressure
Multiplies denial rewards based on the opponent's desperation.
*   **Condition**: `Enemy_Timeout < 25`.
*   **Multiplier**: **2.0x**.
*   **Effect**: Aggression scales up when the opponent is vulnerable to elimination.

---

## 3. Version History
- **v1.0**: Baseline implementation. Simple "Chase" blocking.
- **v2.0 (Tier 1: Glasses)**: Expanded `cp_obs` (10-dim). Solved "Myopic Driver" issue.
- **v2.1 (Tier 2: Wrestler)**: Expanded `entity_obs` (14-dim). Implemented Zone Control & Timeout Pressure.

## 4. Pending / Future
- **TTC (Time To Collision)**: Explicit physics calculation for input (currently implicit).
- **Trajectory Prediction**: explicit ghost inputs for t+3 positions.
