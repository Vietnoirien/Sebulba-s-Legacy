from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class Stage(ABC):
    """
    Abstract Base Class for Curriculum Stages.
    Encapsulates logic for graduation, objectives, and stage-specific configuration.
    """
    def __init__(self, name: str, config: Any):
        self.name = name
        self.config = config # CurriculumConfig

    def check_graduation(self, metrics: Dict[str, Any], env: Any) -> Tuple[bool, str]:
        """Deprecated: Use update() instead."""
        return False, ""

    def update(self, trainer: Any) -> Tuple[Any, str]:
        """
        Updates stage logic (difficulty, logging, checks).
        Returns:
            (next_stage_id, reason) or (None, "")
        """
        return None, ""


    @abstractmethod
    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        """
        Returns a list of calculated objective values for a population member.
        Used by NSGA-II evolution.
        Args:
            p: Population member dictionary
        Returns:
            List of float values (to be maximized or minimized as per implementation)
        """
        pass
        
    @abstractmethod
    def get_env_config(self) -> Any: # Type hint Any to avoid circular import issues if Config not imported here
        """Returns the Environment Configuration for this stage."""
        pass

    @abstractmethod
    def get_active_pods(self) -> List[int]:
        """Returns the list of active pod indices for this stage."""
        pass

    def on_enter(self, env: Any):
        """Called when this stage becomes active."""
        pass

    def update_step_penalty(self, base_penalty: float) -> float:
        """Returns the step penalty for this stage."""
        return base_penalty


    @property
    def target_evolve_interval(self) -> int:
        """Number of iterations between evolution steps."""
        return 2 # Default
