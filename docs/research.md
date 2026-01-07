# Research Context

## 1. Problem Formulation: "Escaping the Gold League"

The **Mad Pod Racing** challenge on CodinGame represents a complex continuous control problem with a specific hurdle known as the "Gold League Plateau".

### The Challenge
*   **State Space**: Continuous tabular data (Positions, Velocities, Angles of Checkpoints/Pods).
*   **Action Space**: Continuous (Thrust: 0-100, Angle: -18 to +18 degrees).
*   **Constraints**:
    *   **Time**: < 150ms per turn response time.
    *   **Size**: Source code must be single-file and under **100,000 characters**.
    *   **Opponents**: Unknown policies (from simple path-followers to complex heuristic searchers).

### The "Gold League" Problem
In the Gold League, agents typically master basic pathfinding (getting to checkpoints). The transition to **Legend League** requires:
1.  **Adversarial Interaction**: actively blocking opponents while racing.
2.  **Long-term Planning**: Deviating from the optimal racing line to secure a delayed advantage (e.g., waiting for a boost).
3.  **Coordination**: In 2v2 modes, sacrificing one pod to ensure the other wins.

Traditional Heuristic Search (minimax/MCTS) struggles here due to the branching factor of continuous physics and the depth required for strategic blocking. Our hypothesis is that **Massive Scale Reinforcement Learning** can discover these emergent behaviors that are too complex to hand-code.

## 2. Methodology: The OpenAI Five Approach

We explicitly adopted the architecture and philosophy of **OpenAI Five** (Dota 2), adapting it to the constraints of the Pod Racer environment.

### "Copying the Stack"
Our system mirrors the core tenets of the OpenAI Five system:
1.  **PPO (Proximal Policy Optimization)**: We use PPO with Generalized Advantage Estimation (GAE) for its stability in continuous control.
2.  **Massive Scale**: Like OpenAI Five, we rely on large batch sizes (Horizon 128 * 8192 Envs = ~1M samples per update) to smooth out the variance of the gradients.
3.  **Self-Play**: The primary training signal comes from beating a copy of oneself (or past selves), creating a natural curriculum where the opponent improves exactly as fast as the agent.
4.  **LSTM / Memory** (Experimental): While our primary submission uses MLP for speed/size, the infrastructure supports LSTM to handle partial observability.

### Formalized Architecture
We define our policy $\pi_\theta(a | s, z)$ where:
*   $s$: The observation state (Self, Enemies, Checkpoints).
*   $z$: A discrete **Role Embedding** ($z \in \{0, 1\}$).
*   $\theta$: The shared parameters of the neural network.

By conditioning the policy on $z$, we approximate a **Mixture of Experts** without the parameter cost.
$$ \text{PilotNet}(s) \rightarrow \text{DrivingSkills} $$
$$ \text{CommanderNet}(\text{DrivingSkills}, z) \rightarrow \text{RoleBehavior} $$

This allows the agent to learn a specialized "Blocker" manifold and a "Runner" manifold that intersect at the "Basic Driving" manifold, maximizing the utility of every parameter in the 100k character limit.

## 3. Evolutionary Strategy (NSGA-II)
To avoid the need for manual hyperparameter tuning (which is expensive at this scale), we wrap the RL process in a **Population Based Training** loop.

We formulate the selection as a Multi-Objective Optimization problem:
$$ \text{Maximize } F(x) = ( f_{win}(x), f_{novelty}(x), f_{efficiency}(x) ) $$

Using **NSGA-II** (Non-dominated Sorting Genetic Algorithm II), we maintain a Pareto Frontier of agents. This preserves agents that might have a lower Win Rate but exhibit highly novel behavior (potential creative strategies) or high racing efficiency (potential speedsters), preventing the population from collapsing into a single, brittle local optimum.
