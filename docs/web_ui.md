# Web Interface & Visualization

The project includes a realtime React-based dashboard for monitoring training and visualizing the agents behavior.

## 1. Dashboard (`localhost:5173`)
The main dashboard provides high-level control over the training session.

### Control Panel
*   **Init Sequence**: Starts the PPO training loop.
*   **Halt**: Pauses training safely.
*   **Snapshot**: Manually saves the current best agent.
*   **Export**: Generates a `submission.py` for CodinGame deployment.

### Live Tuning
You can modify hyperparameters **during training** without restarting.
*   **Physics**: Adjust friction, collision weights, and movement rewards.
*   **Objectives**: Tune the "Team Spirit" blend or the weight of winning vs lap completion.
*   **Transition**: Change the validation thresholds for stage graduation.
> **Note**: Changes applied via "Apply Live" take effect immediately in the next training iteration.

## 2. 3D Visualization
Clicking "3D View" switches to a WebGL renderer (React Three Fiber) for high-fidelity playback.

### Controls
*   **Camera**:
    *   `Left Click + Drag`: Orbit
    *   `Right Click + Drag`: Pan
    *   `Scroll`: Zoom
*   **Playback**:
    *   `Space`: Pause/Resume
    *   `Slider`: Adjust playback speed (0.1x - 2.0x)

### Features
*   **Two-Pass Glass**: Advanced rendering for transparency sorting (Checkpoints).
*   **Instanced Rendering**: Renders hundreds of objects (trees, barriers) efficiently.
*   **Shield Effects**: Visualizes the active shield state with a blue force-field shader.
