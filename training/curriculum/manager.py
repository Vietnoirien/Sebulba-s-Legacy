from typing import Dict, Any, List
from config import CurriculumConfig
from training.curriculum.base import Stage
from training.curriculum.stages import (
    NurseryStage, SoloStage, BlockerStage, RunnerStage, TeamStage, LeagueStage
)
from simulation.env import (
    STAGE_NURSERY, STAGE_SOLO, STAGE_DUEL_FUSED, STAGE_RUNNER, STAGE_TEAM, STAGE_LEAGUE
)

class CurriculumManager:
    """
    Manages the lifecycle of curriculum stages and transitions.
    """
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.stages: Dict[int, Stage] = {
            STAGE_NURSERY: NurseryStage(config),
            STAGE_SOLO: SoloStage(config),
            STAGE_DUEL_FUSED: BlockerStage(config),
            STAGE_RUNNER: RunnerStage(config),
            STAGE_TEAM: TeamStage(config),
            STAGE_LEAGUE: LeagueStage(config)
        }
        self.current_stage_id = STAGE_NURSERY

    @property
    def current_stage(self) -> Stage:
        return self.stages[self.current_stage_id]
        
    def set_stage(self, stage_id: int):
        """Manually force a stage."""
        if stage_id in self.stages:
            self.current_stage_id = stage_id

    def update(self, trainer):
        """
        Called periodically (e.g. start of iteration).
        Delegates to current stage to check status/graduation.
        """
        # Ensure our internal state matches env if modified externally (e.g. by API)
        if trainer.env.curriculum_stage != self.current_stage_id:
             self.current_stage_id = trainer.env.curriculum_stage
             # Force sync config incase only ID changed
             trainer.env.set_stage(self.current_stage_id, self.current_stage.get_env_config())
             
        # Create/Update helper metrics if needed?
        # Stage logic accesses trainer.env.metrics directly.
        
        next_stage_id, reason = self.current_stage.update(trainer)
        
        if next_stage_id is not None:
             self.transition(trainer, next_stage_id, reason)
             
    def transition(self, trainer, next_stage_id: int, reason: str):
         prev_stage = self.current_stage_id
         self.current_stage_id = next_stage_id
         
         # Propagate to Env with Config
         trainer.env.set_stage(next_stage_id, self.current_stage.get_env_config(), reset_env=True)
         
         # Reset Population Metrics on Transition
         # This ensures 'Wins (EMA)' and other stats reflect the new stage immediately.
         for p in trainer.population:
             p['ema_wins'] = 0.0
             p['ema_denials'] = 0.0
             p['ema_blocker_score'] = 0.0
             p['ema_efficiency'] = 999.0
             p['blocker_collisions'] = 0
         
         trainer.log("Metrics Reset: Cleared Population EMA stats for fresh stage start.")
         
         # --- SNAPSHOT & PRESERVATION LOGIC ---
         if prev_stage == STAGE_DUEL_FUSED and next_stage_id == STAGE_RUNNER:
             # Snapshot Strategy: Broadcast Best Blocker Weights
             
             # 1. Find Best Blocker in Population
             # Metric: ema_denials (primary) + ema_blocker_score (tie breaker)
             # 1. Find Best Blocker in Population
             # User Request: Prioritize Denial Rate, then Efficiency (Less Collisions)
             # Sort Key: (ema_denials DESC, blocker_collisions ASC -> -blocker_collisions DESC)
             
             # Sort population
             candidates = sorted(
                 trainer.population, 
                 key=lambda x: (x.get('ema_denials', 0.0), -x.get('blocker_collisions', 0)), 
                 reverse=True
             )
             
             best_agent = candidates[0]
             best_blocker_id = best_agent['id']
             # Score for logging: Denials %
             best_dr = best_agent.get('ema_denials', 0.0) * 100.0
             best_col = best_agent.get('blocker_collisions', 0)
             
             if best_blocker_id != -1:
                 trainer.log(f"*** PRESERVATION: Selected Agent {best_blocker_id} as Best Blocker template (DR {best_dr:.1f}%, Cols {best_col}) ***")
                 trainer.broadcast_blocker_weights(best_blocker_id)
                 
                 # 2. Freeze Blocker Experts for ALL agents
                 for p in trainer.population:
                     p['agent'].freeze_blocker_experts()
                 
                 trainer.log("*** PRESERVATION: Frozen Blocker Experts for Population ***")
             else:
                 trainer.log("WARNING: Could not find Best Blocker to snapshot.")
             
             # Reset Difficulty for Runner Stage (Fair Start)
             trainer.env.bot_difficulty = 0.60
             trainer.log(f"Config Update: Resetting Bot Difficulty to {trainer.env.bot_difficulty:.2f} for Runner Academy.")

         if prev_stage == STAGE_RUNNER and next_stage_id == STAGE_TEAM:
             # Unfreeze Logic
             for p in trainer.population:
                 p['agent'].unfreeze_all()
             trainer.log("*** PRESERVATION: Unfrozen Experts for Team Stage ***")
         
         trainer.log(f"*** Curriculum Transition: {self.stages[prev_stage].name} -> {self.stages[next_stage_id].name} ***")
         trainer.log(f"*** Reason: {reason} ***")
         
         
         
         # Reset Environment handled by Trainer.
         # So we just set the state here.



         
    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        return self.current_stage.get_objectives(p)
        
    def get_active_pods(self) -> List[int]:
        return self.current_stage.get_active_pods()

    def update_step_penalty(self, base_penalty: float) -> float:
        return self.current_stage.update_step_penalty(base_penalty)

    def update_team_spirit(self, trainer) -> float:
        """
        Updates the team_spirit configuration based on stage progress.
        Stage < 4: 0.0
        Stage >= 4: Managed via on_evolution_step (Evolution-based annealing)
        """
        current = trainer.team_spirit
        
        if self.current_stage_id < STAGE_TEAM:
            current = 0.0
        elif self.current_stage_id == STAGE_LEAGUE:
            current = 1.0
            
        return current

    def on_evolution_step(self, trainer):
        """
        Called when population evolution occurs.
        Increments team_spirit by 0.01 (Standard) or 0.05 (Turbo if Difficulty Maxed).
        """
        if self.current_stage_id >= STAGE_TEAM:
            # Dynamic Annealing
            # If Bot Difficulty is Maxed (Ready for Graduation), speed up Spirit Annealing
            grad_diff = self.config.team_graduation_difficulty
            if trainer.env.bot_difficulty >= grad_diff:
                 step = 0.05 # Turbo Spirit: 20 evolutions -> 100%
            else:
                 step = 0.01 # Standard Spirit: 100 evolutions -> 100%
            
            trainer.team_spirit = min(1.0, trainer.team_spirit + step)
