# Sebulba's Legacy (Pod Racer AI)

![Interface Preview](assets/interface_preview.png)
*Real-time training dashboard visualizing 4096 concurrent agents and population metrics.*

## 🏎️ Project Overview

**Sebulba's Legacy** is an advanced Reinforcement Learning (RL) system designed to train super-human autonomous racers for the *Mad Pod Racing* environment. This project replaces the original Sebulba pod trainer, introducing a fully **vectorized environment**, enhanced **model infrastructure**, and refined **Genetic Algorithm** techniques. 

Unlike traditional RL implementations that run a handful of environments, this project leverages a custom **GPU-Accelerated Physics Engine** to simulate **4096 environments in parallel** on a single consumer GPU (achieving **>20,000 Steps Per Second** on an RTX 5070). This massive throughput allows for the training of robust agents using **Population-Based Training (PBT)** within hours rather than days.

The system combines state-of-the-art techniques from Deep Learning and Evolutionary Algorithms to solve complex continuous control problems.

## 🚀 Key Features

### 🧠 Advanced Reinforcement Learning
*   **Massive Parallelism**: Trains on **4096 concurrent environments** using pure PyTorch operations (Sim-to-Tensor), bypassing CPU bottlenecks.
*   **PPO + GAE**: Utilizes Proximal Policy Optimization with Generalized Advantage Estimation for stable and sample-efficient learning.
*   **DeepSets Architecture**: Agents use a **Permutation-Invariant** neural network to handle varying numbers of opponents (Solo, Duel, League) without architecture changes.
*   **Intrinsic Curiosity (RND)**: Incorporates **Random Network Distillation** to encourage exploration in sparse reward scenarios, preventing premature convergence.

### 🧬 Evolutionary Strategy (GA + RL)
*   **Population-Based Training (PBT)**: Evolves a population of 32 distinct agents. Agents don't just learn a policy; they evolve their hyperparameters (Learning Rate, Entropy Coefficient) and reward weights over time.
*   **NSGA-II Selection**: Uses **Non-Dominated Sorting Genetic Algorithm II** to select elite agents based on multiple conflicting objectives:
    *   **Efficiency**: Minimizing steps per checkpoint.
    *   **Consistency**: Maximizing checkpoint streak.
    *   **Novelty**: Maximizing behavioral diversity to prevent population collapse.
*   **Dynamic Reward Shaping**: The system "discovers" the optimal reward function by mutating the weights of various signals (Velocity, Orientation, Winning) during evolution.

### 📈 Curriculum Learning
The training process is automated through distinct stages of difficulty:
1.  **Stage 0: Solo Time Trial**: Agents maximize speed and control to navigate checkpoints (Goal: >50k Laps).
2.  **Stage 1: Duel (1v1)**: Agents face a scripted bot with dynamic difficulty scaling to learn collision avoidance and overtaking.
3.  **Stage 2: League**: Agents compete against a persistent "League" of historical elite agents in full 4-pod races.

### 📊 Real-Time Visualization
*   **Web Dashboard**: A React + Konva frontend rendering the simulation at 60 FPS.
*   **Telemetry**: Displays real-time metrics for Learning Rate, Entropy, Win Rate, and detailed Population Stats.
*   **Interactive Control**: Allows users to manually halt training, export submissions, or "Wipe/Reset" checkpoints dynamically.

## 🛠️ Architecture

```mermaid
graph TD
    subgraph GPU [GPU Acceleration]
        Sim["Vectorized Simulation (4096 Envs)"]
        Physics[Custom Physics Engine]
        Models["Agent Population (DeepSets)"]
    end
    
    subgraph CPU [CPU Orchestration]
        PPO[PPO Trainer]
        GA["Evolutionary Controller (NSGA-II)"]
        API[FastAPI Backend]
    end
    
    subgraph Client [Web Interface]
        React[React Dashboard]
    end
    
    Sim -->|States & Rewards| PPO
    PPO -->|Actions| Sim
    Sim -->|Telemetry| API
    PPO -->|Metrics| GA
    GA -->|Mutated Weights| PPO
    API -->|WebSockets| React
```

## 📦 Installation

### Prerequisites
*   **OS**: Linux (Recommended) or Windows (WSL2)
*   **Python**: 3.12+
*   **Node.js**: 20+
*   **GPU**: NVIDIA GPU with CUDA support (Required for simulation)
    > [!WARNING]
    > **Strict GPU Requirement**: This project relies on a custom GPU-based physics engine (`gpu_physics.py`) for massive parallelism. There is **NO CPU fallback**. Attempting to run this on a CPU-only machine will fail.

### Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Vietnoirien/Sebulba-s-Legacy.git
    cd sebulbas-legacy
    ```

2.  **Install Python Dependencies**

    > [!IMPORTANT]
    > To compute with Torch on an NVIDIA GPU, it must be installed from [pytorch.org](https://pytorch.org/get-started/locally/) first.

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate.ps1 on Windows

    # Example: Install PyTorch with CUDA support (Visit website for your specific command)
    # pip3 install torch --index-url https://download.pytorch.org/whl/cu118

    pip install -r requirements.txt
    pip install aiohttp # For telemetry
    ```

3.  **Install Frontend Dependencies**
    ```bash
    cd web/frontend
    npm install
    cd ../..
    ```

## 🎮 Usage

The project includes a unified launcher to start the Training Loop, Backend API, and Frontend simultaneously.

**Start the System:**
```bash
python launcher.py
```

*   **Dashboard**: Open `http://localhost:5173`
*   **API**: `http://localhost:8000`

### Dashboard Interface (Web UI)

**1. Operations Panel**
*   **Start With...**: Choose to start training from scratch, resume a previous generation, or load a specific checkpoint.
*   **Progression Mode**:
    *   **AUTO**: Automatically advances stages based on graduation thresholds (Efficiency & Consistency).
    *   **MANUAL**: Force a specific stage (Solo, Duel, League).
*   **Stage Selector**: Manually override the current curriculum stage (when in Manual mode).

**2. Control Buttons**
*   **INIT SEQUENCE**: Starts or resumes the training loop with the selected configuration.
*   **HALT**: Safely pauses the training loop (waits for current step to finish).
*   **RESET**: Resets the environment state without clearing the model.
*   **SNAPSHOT**: Manually saves a checkpoint of the current leader.
*   **EXPORT SUBMISSION**: Generates a `submission.py` from the current active model, ready for upload to Codingame.
*   **WIPE ALL CHECKPOINTS**: ⚠️ Destructive action. Deletes all saved models and generations to start fresh.

**3. Hyperparameters (Real-time)**
*   **Learning Rate**: Adjust the PPO learning rate on the fly.
*   **Entropy Coef**: Tune exploration vs. exploitation dynamically.

### Headless / CLI Usage
You can run the components individually without the Launcher/UI:

**1. Training Loop (`ppo.py`)**
Run the training logic directly. Useful for debugging or server environments.
```bash
python -m training.ppo
```

**2. Export Submission (`export.py`)**
Convert a trained model checkpoint into a standalone `submission.py` for Codingame.
```bash
python export.py --model data/generations/gen_X/agent_Y.pt --out submission.py
```

## ⚙️ Configuration

Key configurations can be found in `config.py` and `simulation/env.py`.

*   **Map Size**: 16000 x 9000
*   **Physics**: Large Pod Radius (400), High Friction (0.85).
*   **Reward Function**: 
    *   Velocity uses **Potential-Based Reward Shaping** (Difference of potentials) to guarantee policy invariance.
    *   Orientation uses a cosine alignment metric.
    
## 🏆 Credits

*   **Original Game**: Based on the [Mad Pod Racing](https://www.codingame.com/multiplayer/bot-programming/mad-pod-racing) challenge on Codingame.
*   **Assets**: Original Pod Sprites courtesy of [Codingame](https://www.codingame.com/multiplayer/bot-programming/mad-pod-racing).
*   **Inspiration**: DeepMind's AlphaStar (League Training) and OpenAI's PPO (Proximal Policy Optimization).

---
*Built with ❤️ by Vietnoirien*
