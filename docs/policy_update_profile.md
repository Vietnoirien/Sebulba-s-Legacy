# Policy Update Profile: Split Architecture & Curriculum

This report verifies how the new Split Architecture (Runner/Blocker Experts) interacts with the Training Curriculum. It details which network branches receive gradients in each stage ensuring efficient and correct policy updates.

## Architecture Overview
The `PodActor` now features a Hard-Gated Mixture of Experts (MoE):
- **Input Gating**: `role_id` (0=Runner, 1=Blocker) selects the specific `PilotNet` and `EnemyEncoder` to use.
- **Execution**: The corresponding LSTM/MLP expert is executed.
- **Gradient Flow**: Gradients only flow through the active expert's branch. The inactive branch receives zero gradients for that sample.

## Stage Analysis

### Stage 0: Nursery & Stage 1: Solo
**Configuration:**
- **Active Pods**: `[0]` (Single Agent)
- **Opponents**: None (Inactive/Despawned)
- **Role Assignment**: Dynamic Fallback.
    - Since Pod 1 is inactive (positioned at infinity), its progress score is minimal.
    - Pod 0 (Active) trivially beats Pod 1.
    - Result: `is_runner[0] = True`.

**Policy Update Profile:**
- **Active Role**: **Runner (100%)**
- **Active Branches**:
    - `PilotNet (Runner)`: **Active**
    - `EnemyEncoder (Runner)`: **Active** (Receives inputs, though enemies are far)
    - `Runner Backbone/LSTM`: **Active**
- **Frozen Branches**:
    - `Blocker` components receive **NO Gradients**.
- **Goal**: Pure navigation training for the Runner expert.

---

### Stage 2: Unified Duel
**Configuration:**
- **Active Pods**: `[0, 2]` (Agent vs Bot)
- **Role Assignment**: **Split Batch (Explicit)**
    - `env.py` splits the batch of environments into two halves.
    - **Group A (50%)**: Pod 0 is **Runner**.
    - **Group B (50%)**: Pod 0 is **Blocker**.

**Policy Update Profile:**
- **Active Role**: **Mixed (50% Runner / 50% Blocker)**
- **Active Branches**:
    - **BOTH** Runner and Blocker branches are active within the same PPO batch.
    - Gradients are averaged across the batch, effectively training both experts simultaneously.
- **Goal**: Simultaneous mastery of Racing and Blocking. The "Unified" nature ensures the agent doesn't forget how to race while learning to block.

---

### Stage 3: Team & Stage 4: League
**Configuration:**
- **Active Pods**: `[0, 1, 2, 3]` (2v2)
- **Training Batch**: Multi-Agent (Contains observations from both Pod 0 and Pod 1).
- **Role Assignment**: **Fixed**
    - Pod 0: **Runner**
    - Pod 1: **Blocker**

**Policy Update Profile:**
- **Active Role**: **Mixed (Interleaved)**
    - The batch contains samples from Pod 0 (Runner) and Pod 1 (Blocker).
- **Active Branches**:
    - **BOTH** branches update simultaneously.
- **Goal**: Specialized refinement. Pod 0 refines the Runner expert; Pod 1 refines the Blocker expert.

## Conclusion
The implementation correctly aligns with the curriculum requirements:
1.  **Isolation**: Early stages (Nursery/Solo) exclusively train the Runner expert, preventing "pollution" from Blocker objectives.
2.  **Coverage**: Duel and Team stages ensure full coverage of both experts.
3.  **Safety**: The hard-gating mechanism guarantees that reward signals for specific behaviors (e.g., stopping to block) strictly update the relevant expert, resolving the "Shared Component Interference" issue.
