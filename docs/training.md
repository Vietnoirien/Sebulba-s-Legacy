# Training & Curriculum

## 1. Curriculum Learning
The agent traverses through 6 distinct stages of difficulty, graduating only when specific performance metrics are met.

| Stage | Name | Format | Goal | Graduation |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **Nursery** | Solo (No Enemy) | Learn basic navigation and physics control. | Consistency Score > 500 |
| **1** | **Solo** | Time Trial | Maximize speed and trajectory efficiency. | Standardized Score > 3000 |
| **2** | **Duel** | 1v1 vs Bot | Defeat a dynamic scripted opponent. | Win Rate > 65% (vs "Insane" Bot) |
| **3** | **Intercept** | 2v2 (Blocker vs Runner) | **Blocker Academy**: Learn to deny the opponent Runner. | Denial Rate > 60% |
| **4** | **Team** | 2v2 vs Team | Cooperative racing with Runner/Blocker roles. | Win Rate > 60% |
| **5** | **League** | 2v2 vs History | Full league play against past elite agents. | N/A (End Game) |

### Transition to Team Play
Entering Stage 3 (Intercept) introduces role specialization.
*   **Blocker Academy**: The agent controls a single specialized "Blocker" pod. The goal is purely to prevent the opponent Runner from finishing. This teaches collision physics and aggressive blocking without the distraction of trying to race.
*   **Team Mode (Stage 4)**: The agent controls **two pods** simultaneously:
    *   **Pod 0**: Runner (Goal: Win race)
    *   **Pod 1**: Blocker (Goal: Obstruct enemies)
    *   **Team Spirit**: A curriculum-annealed coefficient blends individual selfish rewards with the team's collective outcome, ensuring the Blocker acts in the Runner's best interest.

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
