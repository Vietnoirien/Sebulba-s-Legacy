# Research Report: Trajectory Prediction & Neural-Guided Search

## 1. Executive Summary

Trajectory Prediction is a critical capability for "Legend League" performance in Mad Pod Racing. It shifts the agent from **reactive** ("avoid the pod right in front of me") to **proactive** ("intercept the path the enemy wants to take").

For our **Sim-to-Tensor** architecture and **Neural-Guided Local Search**, we recommend a **Physics-Informed Action Prediction** approach. Instead of predicting raw coordinates (which ignores collisions), we predict the *intent* (Action: Thrust/Angle) of opponents and use our embedded physics engine to resolve the outcome.

**Complexity**: Medium (Requires auxiliary training objectives).
**Benefit**: High (Enables "checkmate" blocking and precise drifting).

---

## 2. Comparative Review: SOTA Approaches

We analyzed three State-of-the-Art paradigms in Multi-Agent Trajectory Prediction and evaluated them against our 100k-character / 150ms constraints.

### A. Social LSTM (Alahi et al.) / Social GAN
*   **Concept**: Uses a "Social Pooling" layer where efficient grids share hidden states between neighboring agents (LSTMs).
*   **Pros**: Explicitly models "I move because you moved".
*   **Cons**: High computational overhead ($O(N^2)$ interactions). Hard to export to a simple Python list-based inference engine.
*   **Verdict**: **Too Heavy**. The custom pooling logic consumes too much of the 100k character budget.

### B. Trajectron++ (Salzmann et al.)
*   **Concept**: A graph-structured model that incorporates **heterogeneous data** (Map + Dynamic Constraints). It treats agents as nodes in a spatiotemporal graph. It outputs a distribution of future trajectories (GMM).
*   **Pros**: Respects physical constraints (non-holonomic robots). Handles map context (walls/checkpoints).
*   **Cons**: Graph Neural Networks (GNNs) are complex to implement in a "Zero-Dependency" submission.
*   **Verdict**: **Best Theoretical Fit**, but impractical export. We should borrow the "Map-Awareness" but skip the GNN.

### C. World Models (Ha & Schmidhuber) / MuZero
*   **Concept**: Learn a full internal simulation model $P(S_{t+1} | S_t, A_t)$. The agent "dreams" potential futures.
*   **Pros**: Universal. Can predict complex bounces and friction.
*   **Cons**: Extremely expensive to train and run.
*   **Verdict**: **Redundant**. We *already* have a World Model: the game rules are known and deterministic. Using a neural approximation of a known physics engine is wasteful. Use the actual physics engine (S2).

### D. Proposed: "Physics-Informed Action Prediction" (Hybrid)
*   **Concept**: Instead of predicting $Pos_{t+1}$ (which is hard to learn perfectly due to collisions), predict $\hat{A}_{opponent}$ (Enemy Intent). Then compute $Pos_{t+1} = \text{Physics}(Pos_t, \hat{A}_{opponent})$.
*   **Pros**: 100% physically valid predictions. Extremely lightweight (just small MLP heads).
*   **Cons**: Depends on accurate physics implementation (which we have).

---

## 3. Architecture Integration

We propose integrating **Trajectory Prediction** into the existing **Recurrent Split Backbone** without massive structural changes.

### A. The "Oracle" Head (Auxiliary Task)
During training, the agent has access to the *true* actions taken by enemies (from the batch data). We add an auxiliary loss:
$$ L_{pred} = MSE(\hat{A}_{enemy}, A_{enemy}^{ground\_truth}) $$
*   **Inputs**: The existing `DeepSets` enemy encoding already captures relative state.
*   **Output**: A new head on the `EnemyEncoder` (before pooling) that outputs predicted `[Thrust, Angle]` for each enemy.
*   **Inference**: This gives us a vector $[\hat{T}, \hat{\alpha}]$ for every visible pod.

### B. "Ghost" Observations
We can optionally feed these predictions *back* into the Commander Stream as "Ghost Inputs" (e.g., "Where will the enemy be in $t+5$?").
*   **Implementation**: Run the predicted action through the internal physics model (in the forward pass) to generate `GhostPosition`.
*   **Benefit**: The LSTM Core learns to react to the *future* state.

---

## 4. Synergy with Neural-Guided Local Search

This is the most powerful unlock. Currently, our **System 2 (Local Search)** assumes enemies are static or linear. This fails in "Dogfights" where enemies turn sharply.

**Current Workflow:**
1.  **Generate**: My Candidates $\{C_1, C_2, ...\}$
2.  **Simulate**: $S_{t+1} = \text{Phys}(S_t, C_i, \text{Enemy}_{static})$
3.  **Score**: Evaluation assumes enemies didn't react.

**New "Clairvoyant" Workflow:**
1.  **Predict**: $\hat{A}_{enemy} = \text{Policy}_{pred}(\text{Observation})$
2.  **Generate**: My Candidates $\{C_1, C_2, ...\}$
3.  **Simulate**: $S_{t+1} = \text{Phys}(S_t, C_i, \hat{A}_{enemy})$
    *   *Note*: The physics engine now simulates the **interaction** (collision) between my candidate move and the enemy's predicted move.
4.  **Score**: We can now score based on "Successful Blocking".
    *   Did I crash into him? (Good if I am blocker).
    *   Did I cut off his line?

### Complexity Analysis
*   **Training**: Negligible. Unsupervised auxiliary task.
*   **Export Size**: Very low. Just one small MLP head (e.g., $64 \to 2$) per enemy. ~130 params.
*   **Inference Latency**:
    *   Current: 1 Step Sim for Ego (Cheap).
    *   New: 1 Step Sim for Ego + Enemies.
    *   The `N` class physics engine is vector-capable or fast loops. simulating 3 extra pods per candidate is well within the 1000ms (we currently use ~10ms).

## 5. Recommendation

**Implement "Action Prediction" immediately.**

1.  **Add `pred_head`** to the `EnemyEncoder` in `architecture.py`.
2.  **Add `L_pred`** to the loss function in `ppo.py`.
3.  **Update `export.py`** to include these weights.
4.  **Update `submission.py`** (System 2) to use these predictions during the simulation step.
