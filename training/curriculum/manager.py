from typing import Dict, Any, List
from config import CurriculumConfig
from training.curriculum.base import Stage
from training.curriculum.stages import (
    NurseryStage, SoloStage, DuelStage, InterceptStage, TeamStage, LeagueStage
)
from simulation.env import (
    STAGE_NURSERY, STAGE_SOLO, STAGE_DUEL, STAGE_INTERCEPT, STAGE_TEAM, STAGE_LEAGUE
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
            STAGE_DUEL: DuelStage(config),
            STAGE_INTERCEPT: InterceptStage(config),
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
         
         trainer.log(f"*** Curriculum Transition: {self.stages[prev_stage].name} -> {self.stages[next_stage_id].name} ***")
         trainer.log(f"*** Reason: {reason} ***")
         
         # Reset Environment or trigger Mitosis handled by Trainer?
         # Trainer checks (prev != current) loop start and resets.
         # So we just set the state here.
         
         if next_stage_id == STAGE_TEAM:
             # self.perform_mitosis(trainer) # DEPRECATED: Universal Actor shares weights.
             pass



         
    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        return self.current_stage.get_objectives(p)
        
    def get_active_pods(self) -> List[int]:
        return self.current_stage.get_active_pods()

    def update_step_penalty(self, base_penalty: float) -> float:
        return self.current_stage.update_step_penalty(base_penalty)
