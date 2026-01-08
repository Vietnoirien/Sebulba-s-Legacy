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
In the Gold League, agents typically master basic pathfinding. The transition to **Legend League** requires:
1.  **Adversarial Interaction**: actively blocking opponents while racing.
2.  **Long-term Planning**: Deviating from the optimal racing line to secure a delayed advantage.
3.  **Coordination**: In 2v2 modes, sacrificing one pod to ensure the other wins.

Traditional Heuristic Search (minimax/MCTS) struggles here due to the branching factor of continuous physics. Our hypothesis is that **Massive Scale Reinforcement Learning** can discover these emergent behaviors.

## 2. Methodology: The "Sim-to-Tensor" Framework

We explicitly adopted the architecture of **OpenAI Five**, but adapted it for consumer hardware by keeping the entire lifecycle on the GPU.

### Vectorized Optimization
Standard PPO implementations suffer from CPU-GPU bandwidth bottlenecks. We implement a **fully vectorized training loop**:
1.  **GPU Physics**: The simulation step `S_t+1 = Sim(S_t, A_t)` is a PyTorch tensor operation.
2.  **Vectorized Adam**: Instead of maintaining a list of 128 `torch.optim.Adam` instances (one per agent in the population), we implement a single custom optimizer that treats the population dimension as a batch dimension:
    *   Parameters: `[Population, Layers, Features]`
    *   Operations: `param -= lr * grad / (sqrt(v) + eps)` applied via `vmap`.
    This allows us to train **64 distinct policies** simultaneously with the throughput of a single large batch interaction.

## 3. Architecture: Recurrent Split Backbone

To solve the "100k Character Limit" constraint while retaining complex multi-modal behavior (Racing vs Blocking), we utilize a **True Parameter Sharing** architecture with a Recurrent Core.

### The "Universal Brain" (~58k Parameters)
We define our policy $\pi_\theta(a | s, z)$ where $z \in \{0, 1\}$ is a **Role Embedding**.
*   **Pilot Stream**: A lightweight MLP ($25 \rightarrow 64 \rightarrow 48$) processes "Reactive" features (Self Vector + Next Checkpoint).
*   **Commander Stream**: A structured path ($95 \rightarrow 96 \rightarrow 48$) processes "Tactical" features (Teammates, Enemies, Map embedding).
    *   **DeepSets**: Enemies are processed via a permutation-invariant encoder $g(x) = \max_i(f(x_i))$.
    *   **Map Transformer**: A **Nano-Transformer** (1 layer, 2 heads, $d_{model}=32, ~8.6k$ params) encodes the sequence of future checkpoints via Self-Attention to provide track foresight.
*   **LSTM Core**: Both streams fuse into an LSTM ($H=48, ~28k$ params), allowing the agent to maintain temporal context (e.g., "I just crashed", "I am waiting for a boost").

By conditioning the Commander Stream on $z$, the network learns two distinct behavioral manifolds (Runner/Blocker) that share the same underlying physics understanding (Pilot Stream).

## 4. Hybrid Inference: Neural-Guided Local Search

Pure RL policies can be jittery or miss precise geometric opportunities. We implement a **System 1 / System 2** hybrid inference:

1.  **System 1 (Neural Net)**: The PPO policy $\pi_\theta(s)$ proposes a distribution of actions. We sample the mode (greedy) and several nearby variations ($\pm 3^\circ, \pm 6^\circ$).
2.  **System 2 (Local Search)**: We use a **minified physics engine** (embedded in the submission) to simulate these candidate actions for 1 step.
    *   $a^* = \text{argmax}_{a \in \mathcal{C}} \text{Scoring}(\text{Sim}(s, a))$
    *   The scoring function penalizes collisions and rewards alignment with the racing line.

This "Safety Layer" allows the RL to be aggressive/creative while the Local Search handles the fine-grained precision required to pass checkpoints without collision.

## 5. Evolutionary Strategy (NSGA-II)
To avoid manual hyperparameter tuning, we wrap the RL process in a **Population Based Training** loop using **NSGA-II** selection.
$$ \text{Maximize } F(x) = ( f_{win}(x), f_{novelty}(x), f_{efficiency}(x) ) $$
This preserves agents that might have a lower Win Rate but exhibit highly novel behavior, preventing the population from collapsing into a specific meta.
