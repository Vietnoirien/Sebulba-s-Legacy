# Training & Curriculum

## 1. Recursive Curriculum Learning
The agent traverses through 5 distinct stages of difficulty. Each stage focuses on a specific behavioral primitive, ensuring the agent learns to "walk before it runs."

| Stage | Name | Format | Focus |
| :--- | :--- | :--- | :--- |
| **0** | **Nursery** | Solo (No Enemy) | **Basic Control**. Learning the mapping between Thrust/Angle and velocity vectors. |
| **1** | **Solo** | Time Trial | **Optimization**. Finding racing lines that minimize distance and maximize speed. |
| **2** | **Duel** | 1v1 vs Bot | **Combat Intro**. Learning to race while accounting for a dynamic obstacle (the opponent). |
| **3** | **Intercept** | 1v1 (Blocker Role) | **Aggression**. The "Blocker Academy". The agent MUST deny the opponent from finishing. |
| **4** | **Team** | 2v2 (Runner+Blocker) | **Coordination**. Controlling two pods simultaneously to maximize Team Score. |
| **5** | **League** | 2v2 vs History | **Robustness**. Meta-gaming against the entire history of policies. |

### Graduation Criteria (The Quality Gates)
To prevent "luck-based" progression, we use strict statistical gates:

*   **Nursery $\to$ Solo**: `Consistency > 500`. The agent must finish the track without crashing into walls for 100 consecutive episodes.
*   **Solo $\to$ Duel**: `Efficiency < 40.0`. The agent must complete checkpoints with near-optimal movement.
*   **Duel $\to$ Intercept**: `Win Rate > 65%` against the "Insane" scripted bot.
*   **Intercept $\to$ Team**: `Denial Rate > 60%`. The Blocker must successfully prevent the opponent runner from finishing the race.
*   **Team $\to$ League**: `Win Rate > 60%` against a coordinated team of scripted bots.

## 2. Population Based Training (PBT)
We do not train a single agent. We evolve a population of **128 Agents**.
*   **Hyperparameter Evolution**: Agents mutate their Learning Rate, Entropy Coefficient, and Clip Range during training.
*   **Selection**: Every `N` updates, the bottom 25% of the population is replaced by mutated clones of the top 25%.
*   **Objectives**: Selection is based on **NSGA-II** (Non-Dominated Sorting), balancing multiple conflicting goals:
    1.  **Win Rate** (Performance)
    2.  **Behavioral Novelty** (Diversity)
    3.  **Efficiency** (Steps per checkpoint)

## 3. The League (PFSP)
In Stage 5, agents stop fighting scripted bots and begin fighting **Historical Checkpoints**.
*   **Prioritized Fictitious Self-Play**: We maintain a pool of past checkpoints.
*   **Matchmaking**: Opponents are selected based on **Regret** (Win Rate ~50%). Agents play against opponents they struggle against, not just the random history.
*   **Exploiters**: The population includes "Exploiter" agents designed solely to beat the current leader, preventing the main population from over-fitting to a specific meta.

## 4. Optimization & Resources
The addition of the **Map Transformer** increased the memory footprint of the agent. To maintain high-throughput training (8192 environments) on consumer GPUs (e.g., RTX 5070 12GB), we adjust the PPO batching strategy:
*   **Sequential Minibatches**: We increased `num_minibatches` from 64 to **128**. This halves the memory required for storing activations during the PPO backward pass, preventing `CUDA out of memory` errors without reducing the total number of environments or simulation throughput.
*   **Math Attention**: We explicitly enforce `SDPBackend.MATH` for the Map Transformer to ensure compatibility with `torch.vmap` during the vectorized forward pass.
