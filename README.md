# Sebulba's Legacy (Pod Racer AI)

![Interface Preview](assets/interface_preview.png)
*Training 8,192 agents in parallel at 66,000 steps per second.*

## 🏎️ The 66k SPS Super-Learner

**Sebulba's Legacy** is a massive-scale Reinforcement Learning system inspired by **OpenAI Five**, designed to solve the *Mad Pod Racing* challenge on CodinGame.

By moving the entire simulation and training pipeline to the GPU (**Sim-to-Tensor**), we bypass Python's CPU bottlenecks, achieving **~66,000 Steps Per Second** on a single consumer GPU (RTX 5070). This allows us to train 8,192 environments in parallel, evolving complex cooperative strategies in hours rather than weeks.

### 📚 Documentation
*   [**Architecture**](docs/architecture.md): The System Design, Split Backbone, and PPO implementation.
*   [**Simulation**](docs/simulation.md): GPU Physics engine, Predefined Maps, and Constraints.
*   [**Training**](docs/training.md): Curriculum Stages, PBT, and League mechanics.
*   [**Research**](docs/research.md): Problem formulation and scholarly context.
*   [**Web UI**](docs/web_ui.md): Dashboard and 3D Visualization guide.

---

## 🚀 Key Features

*   **Vectorized Everything**: From simulation to optimizer (`VectorizedAdam`), everything stays on VRAM.
*   **Split Backbone Actor**: A single policy network drives both "Runner" and "Blocker" roles using learned Role Embeddings keying unified driving skills.
*   **Population Based Training (PBT)**: Evolves hyperparameters (LR, Entropy) alongside the policy using NSGA-II selection.
*   **League Play**: Agents train against a prioritized history of past selves (PFSP) to prevent cyclical strategy collapse.

## 📦 Quick Start

### Prerequisites
*   **GPU**: NVIDIA RTX 3000+ (Required for CUDA simulation).
*   **OS**: Linux (Recommended) or WSL2.
*   **Python**: 3.12+

### Installation
1.  **Clone & Setup**
    ```bash
    git clone https://github.com/Vietnoirien/Sebulba-s-Legacy.git
    cd sebulbas-legacy
    python -m venv .venv
    source .venv/bin/activate
    
    # Install Torch (Check pytorch.org for your CUDA version)
    pip3 install torch --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

2.  **Install Frontend**
    ```bash
    cd web/frontend && npm install && cd ../..
    ```

### Usage
Start the integrated Launcher (Backend + Frontend + Training Loop):
```bash
python launcher.py
```
Open **[http://localhost:5173](http://localhost:5173)** to access the dashboard.

## 🏆 Credits
*   **Inspiration**: [OpenAI Five](https://openai.com/research/openai-five) (Architecture & PPO), [DeepMind AlphaStar](https://deepmind.google/discover/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/) (League).
*   **Logic**: Based on the [Mad Pod Racing](https://www.codingame.com/multiplayer/bot-programming/mad-pod-racing) challenge.
*   **3D Models**: [Willson Weber](https://willsonweber.com/).

---
*Built with ❤️ by Vietnoirien*
