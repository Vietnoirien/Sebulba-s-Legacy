# Project Tasks: Sebulba's Legacy

## Phase 1: Core Upgrades (Current Focus)
- [x] **Model Enhancement**
    - [x] **Architecture**: Upscale `DeepSets` hidden dimensions (e.g., 64 -> 256) to utilize 100KB budget.
    - [x] **Dynamic Export**: Refactor `export.py` to dynamically inspect model layers instead of using hardcoded lists.
    - [x] **Template**: Update `SINGLE_FILE_TEMPLATE` in `export.py` to write model dimensions dynamically (avoid hardcoded `13*32`).

- [x] **Export System**
    - [x] **Backend**: Implement `POST /api/export` endpoint in `web/backend/main.py` to trigger `export.py` and return the file content or path.
    - [x] **Frontend**: Add "EXPORT SUBMISSION" button to `ControlPanel.tsx` that calls the API and lets user download `submission.py`.
    - [x] **Validation**: Create `tests/test_export_run.py` to verify the exported script runs and produces valid output matching the model.

## Phase 2: Training Pipeline Gaps
- [ ] **League & ELO System**
    - [ ] **ELO Updates**: Implement `update_elo` calls in `training/ppo.py` when a match (env) finishes. Currently ELO is static.
    - [ ] **Opponent Sampling**: Verify `league.sample_opponent()` correctly feeds historical agents into Team 1 slots.

- [ ] **Telemetry & Visualization**
    - [ ] **Fix Hardcoded Telemetry**: Update `ppo.py`'s `send_telemetry` to send REAL keys (Shield, Boost, Role) from `env.physics` and `env.is_runner`, instead of hardcoded `{{boost:1, shield:0}}`.
    - [ ] **Role Rendering**: Update `RaceCanvas.tsx` to visualize "Runner" (Speed/Flag) vs "Blocker" (Shield/Fist) based on telemetry.

- [ ] **Curriculum Integration**
    - [ ] **Duel Bot**: Verify `STAGE_DUEL` simple bot logic in `env.py` is challenging enough.
    - [ ] **Reward Tuning**: Validate "Team Spirit" (shared reward) works or if we need `beta` blending for individual credits.

## Phase 3: Validation & Polish
- [ ] **Code Cleanup**
    - [ ] **Config**: Move hardcoded physics constants from `export.py` and `simulation/env.py` to `config.py`.
    - [ ] **Type Safety**: Add proper TypeScript interfaces for the expanded Telemetry types (`RaceState`, `Pod`).

- [/] **Final Verification**
    - [x] **Size Check**: Ensure `submission.py` < 100KB with the upscaled model.
    - [x] **Parity Check**: Verify Python `submission.py` logic matches `GPUPhysics` exactly (using `tests/test_physics_parity.py`).
