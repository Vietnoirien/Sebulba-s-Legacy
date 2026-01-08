# Training & Curriculum

## 1. Recursive Curriculum Learning
The agent traverses through 5 distinct stages of difficulty. Each stage focuses on a specific behavioral primitive, ensuring the agent learns to "walk before it runs."

| Stage | Name | Format | Focus |
| :--- | :--- | :--- | :--- |
| **0** | **Nursery** | Solo | **Basic Control**. Learning mapping between Thrust/Angle and velocity. |
| **1** | **Solo** | Time Trial | **Optimization**. Finding racing lines and maximizing speed. |
| **2** | **Unified Duel** | 1v1 vs Bot | **Friction**. Hybrid stage learning to race (Runner) and deny (Blocker) in the same environment. |
| **3** | **Team** | 2v2 (1+1) | **Coordination**. Team of Agent Runner + Agent Blocker vs Bots. Uses a **Tau Safety Net**. |
| **4** | **League** | 2v2 vs History | **Robustness**. Competitive play against a pool of past agent checkpoints. |

## 2. Reward Annealing (Tau)
To ensure a smooth transition from Dense (Instructional) rewards to Sparse (Competitive) rewards, we use the `tau` schedule. 
*   **Stage 0 (Nursery)**: `tau = 0.0` (100% Dense). Learning to navigate.
*   **Stage 1 (Solo)**: `tau = 0.25` (75% Dense). Introducing speed optimization.
*   **Stage 2 (Unified Duel)**: `tau = 0.5` (50% Dense). Racing and Blocking training.
*   **Stage 3 (Team)**: `tau = 0.5` (50% Dense). Team win signals introduced.
*   **Stage 4 (League)**: `tau = 0.85` (15% Dense). Near-pure sparse environment.

All dense rewards (including Blocker-specific rewards like Collision Damage and Denial) are scaled by `(1.0 - tau)`.

### Graduation Criteria (The Quality Gates)
To prevent "luck-based" progression, we use strict statistical gates with **Role-Separated Metrics**:

*   **Nursery $\to$ Solo**: `Consistency > 500`. Finish without crashing for 100 consecutive episodes.
*   **Solo $\to$ Unified Duel**: `Efficiency < 40.0`, `Consistency > 3000`, and `Win Rate > 90%`.
*   **Unified Duel $\to$ Team**: `Bot Difficulty > 0.85` and `Win Rate > 70%` (for 5 consecutive checks) AND `Denial Rate > 5%` (or high Impact).
*   **Team $\to$ League**: `Bot Difficulty > 0.85` and `Win Rate > 70%` (for 5 consecutive checks).

> [!IMPORTANT]
> **Denominator Separation**: We track `Runner_Matches` and `Blocker_Matches` separately. A Runner win does not dilute the Denial Rate, and a Blocker denial does not dilute the Win Rate. This ensures accurate performance tracking in mixed-role stages like Unified Duel.

## 2. Population Based Training (PBT)
We do not train a single agent. We evolve a population of **64 Agents**.
*   **Hyperparameter Evolution**: Agents mutate their Learning Rate, Entropy Coefficient, and Clip Range during training.
*   **Selection (Stage-Based)**: Culling and cloning occur at dynamic intervals depending on the complexity of the current stage:
    *   **Nursery**: Every `1` iteration (Rapid evolution for basic control).
    *   **Solo**: Every `2` iterations (Standard).
    *   **Unified Duel / Team**: Every `5` iterations (Stable evolution required for strategy emergence).
*   **NSGA-II Objectives**: Selection balances multiple conflicting goals:
    1.  **Win Rate** (`Wins / Runner_Matches`)
    2.  **Denial Rate** (`Denials / Blocker_Matches`)
    3.  **Behavioral Novelty**
    4.  **Efficiency** (Steps per checkpoint)

## 3. The League (PFSP)
In Stage 4, agents stop fighting scripted bots and begin fighting **Historical Checkpoints**.
*   **Prioritized Fictitious Self-Play**: We maintain a pool of past checkpoints.
*   **Matchmaking**: Opponents are selected based on **Regret** (Win Rate ~50%). Agents play against opponents they struggle against, not just the random history.
*   **Exploiters**: The population includes "Exploiter" agents designed solely to beat the current leader, preventing the main population from over-fitting to a specific meta.

## 4. Optimization & Resources
The addition of the **Map Transformer** increased the memory footprint of the agent. To maintain high-throughput training (8192 environments) on consumer GPUs (e.g., RTX 5070 12GB), we adjust the PPO batching strategy:
*   **Sequential Minibatches**: We increased `num_minibatches` from 64 to **128**. This halves the memory required for storing activations during the PPO backward pass, preventing `CUDA out of memory` errors without reducing the total number of environments or simulation throughput.
*   **Math Attention**: We explicitly enforce `SDPBackend.MATH` for the Map Transformer to ensure compatibility with `torch.vmap` during the vectorized forward pass.
