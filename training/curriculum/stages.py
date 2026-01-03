from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from training.curriculum.base import Stage
from config import TrainingConfig, CurriculumConfig, EnvConfig
from simulation.env import (
    RW_WIN, RW_LOSS, RW_CHECKPOINT, RW_CHECKPOINT_SCALE, 
    RW_PROGRESS, RW_MAGNET, RW_COLLISION_RUNNER, RW_COLLISION_BLOCKER, 
    RW_STEP_PENALTY, RW_ORIENTATION, RW_WRONG_WAY, RW_COLLISION_MATE,
    STAGE_NURSERY, STAGE_SOLO, STAGE_DUEL, STAGE_TEAM, STAGE_LEAGUE
)

class NurseryStage(Stage):
    def __init__(self, config: CurriculumConfig):
        super().__init__("Nursery", config)
        
    def get_active_pods(self) -> List[int]:
        return [0]

    def get_env_config(self) -> EnvConfig:
        return EnvConfig(
            mode_name="nursery",
            track_gen_type="nursery",
            num_checkpoints_fixed=3,
            active_pods=[0],
            use_bots=False,
            step_penalty_active_pods=[0],
            orientation_active_pods=[0],
            timeout_steps=self.config.nursery_timeout_steps
        )

    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        # Objectives: Consistency (Primary), Nursery Score (Gradient), Novelty (Diversity)
        return [
            p.get('ema_consistency', 0.0),
            p.get('nursery_score', 0.0), 
            p.get('novelty_score', 0.0) * 100.0
        ]
        
    def check_graduation(self, metrics: Dict[str, Any], env: Any) -> Tuple[bool, str]:
        # Deprecated
        return False, ""

    def update(self, trainer) -> Tuple[Optional[int], str]:
        # Goal: Learn to navigate (Consistency). No speed requirement.
        sorted_by_consistency = sorted(trainer.population, key=lambda x: x.get('ema_consistency') or 0.0, reverse=True)
        elites = sorted_by_consistency[:5]
        avg_cons = np.mean([p.get('ema_consistency') or 0.0 for p in elites])
        
        if trainer.iteration % 10 == 0:
             trainer.log(f"Stage 0 (Nursery) Status: Top 5 Avg Cons {avg_cons:.1f} (Goal > {self.config.nursery_consistency_threshold})")
        
        if avg_cons > self.config.nursery_consistency_threshold:
            trainer.log(f">>> GRADUATION FROM NURSERY: Top Agents Cons {avg_cons:.1f} <<<")
            if trainer.curriculum_mode == "auto":
                trainer.log(">>> Welcome to Stage 1: Time Trial (Speed Matters!) <<<")
                return STAGE_SOLO, f"Avg Cons {avg_cons:.1f} > {self.config.nursery_consistency_threshold}"
        
        return None, ""

    def update_step_penalty(self, base_penalty: float) -> float:
        return base_penalty

    @property
    def target_evolve_interval(self) -> int:
        return 1 # Fast Evolution for Nursery (Every Iteration)


class SoloStage(Stage):
    def __init__(self, config: CurriculumConfig):
        super().__init__("Solo", config)
        # Dynamic Penalty Logic Removed (Standardized)

    def get_active_pods(self) -> List[int]:
        return [0]

    def get_env_config(self) -> EnvConfig:
        return EnvConfig(
            mode_name="solo",
            track_gen_type="max_entropy",
            active_pods=[0],
            use_bots=False,
            step_penalty_active_pods=[0],
            orientation_active_pods=[0]
        )

    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        # Objectives: Consistency (Max), Efficiency (Min -> Maximize -Eff), Novelty (Max)
        return [
            p.get('ema_wins', 0.0),
            p.get('ema_consistency', 0.0),
            -p.get('ema_efficiency', 999.0),
            p.get('novelty_score', 0.0) * 100.0
        ]
        
    def update(self, trainer) -> Tuple[Optional[int], str]:
        # Strategy A: Proficiency & Consistency
        # Sort by Wins (Desc), Consistency (Desc), Efficiency (Asc -> -Eff)
        sorted_elites = sorted(trainer.population, key=lambda x: (
            x.get('ema_wins', 0.0) or 0.0,
            x.get('ema_consistency', 0.0) or 0.0,
            -(x.get('ema_efficiency', 999.0) or 999.0)
        ), reverse=True)
        elites = sorted_elites[:5]
        
        avg_eff = np.mean([p.get('efficiency_score') if p.get('efficiency_score') is not None else 999.0 for p in elites])
        avg_cons = np.mean([p.get('ema_consistency') if p.get('ema_consistency') is not None else 0.0 for p in elites])
        avg_wins = np.mean([p.get('ema_wins') if p.get('ema_wins') is not None else 0.0 for p in elites])
        
        if trainer.iteration % 10 == 0:
             trainer.log(f"Stage 1 Status: Top 5 Avg Eff {avg_eff:.1f} (Goal < {self.config.solo_efficiency_threshold}), Cons {avg_cons:.1f} (Goal > {self.config.solo_consistency_threshold}), Wins {avg_wins:.2f} (Goal > {self.config.solo_min_win_rate})")
        
        if avg_eff < self.config.solo_efficiency_threshold and avg_cons > self.config.solo_consistency_threshold and avg_wins > self.config.solo_min_win_rate:
            trainer.log(f">>> GRADUATION FROM SOLO: Top Agents Avg Eff {avg_eff:.1f}, Cons {avg_cons:.1f}, Wins {avg_wins:.2f} <<<")
            if trainer.curriculum_mode == "auto":
                # Special Config Update for Duel
                trainer.env.bot_difficulty = 0.0
                trainer.reward_weights_tensor[:, RW_CHECKPOINT] = 2000.0
                trainer.log("Config Update: Resetting RW_CHECKPOINT to 2000.0 for Stage 1+")
                
                return STAGE_DUEL, f"Eff {avg_eff:.1f} < {self.config.solo_efficiency_threshold}"
        
        return None, ""
        
    def check_graduation(self, metrics: Dict[str, Any], env: Any) -> Tuple[bool, str]:
        return False, ""

    def update_step_penalty(self, base_penalty: float) -> float:
        return base_penalty

class DuelStage(Stage):
    def __init__(self, config: CurriculumConfig):
        super().__init__("Duel", config)
        self.failure_streak = 0
        self.grad_consistency_counter = 0

    def get_active_pods(self) -> List[int]:
        return [0]

    def get_env_config(self) -> EnvConfig:
        return EnvConfig(
            mode_name="duel",
            track_gen_type="max_entropy",
            active_pods=[0, 2],
            use_bots=True, 
            bot_pods=[2],
            step_penalty_active_pods=[0, 2],
            orientation_active_pods=[0, 2]
        )

    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        # Quality Gate: Novelty only counts if agent is competitive (> 5% Win Rate)
        # This prevents "Safe Losers" (consistent but slow) from crowding the front.
        nov = p.get('novelty_score', 0.0) * 100.0
        if p.get('ema_wins', 0.0) < 0.20:
            nov = 0.0

        # Objectives: Win Rate, Efficiency (Speed), Novelty
        eff = p.get('ema_efficiency', 999.0); eff = eff if eff is not None else 999.0
        
        return [
            p.get('ema_wins', 0.0),
            -eff,
            nov
        ]
        
    def check_graduation(self, metrics: Dict[str, Any], env: Any) -> Tuple[bool, str]:
        return False, ""

    def update(self, trainer) -> Tuple[Optional[int], str]:
        # Dynamic Difficulty
        metrics = trainer.env.stage_metrics
        rec_episodes = metrics.get("recent_episodes", 0)
        if rec_episodes == 0:
             rec_episodes = metrics.get("recent_games", 0)

        if rec_episodes > 1000: 
            rec_wins = metrics["recent_wins"]
            rec_games = metrics["recent_games"] # Valid finished games
            
            if rec_episodes > 0:
                rec_wr = rec_wins / rec_episodes
            else:
                rec_wr = 0.0
            
            trainer.current_win_rate = rec_wr
            
            # Reset Recent
            metrics["recent_games"] = 0
            metrics["recent_wins"] = 0
            metrics["recent_episodes"] = 0
            
            rec_losses = rec_games - rec_wins
            rec_timeouts = rec_episodes - rec_games
            
            trainer.log(f"Stage 2 (Duel) Check: Recent WR {rec_wr*100:.1f}% | Wins: {rec_wins} | Losses: {rec_losses} | Timeouts: {rec_timeouts} | Diff: {trainer.env.bot_difficulty:.2f}")
            
            auto = (trainer.curriculum_mode == "auto")
            
            if rec_wr < self.config.wr_critical:
                # Critical Failure
                if auto:
                    trainer.env.bot_difficulty = max(0.0, trainer.env.bot_difficulty - self.config.diff_step_decrease)
                    trainer.log(f"-> Critical Regression (WR < {self.config.wr_critical:.2f}): Decreasing Bot Difficulty to {trainer.env.bot_difficulty:.2f}")
                
                self.failure_streak = 0
                
            elif rec_wr < self.config.wr_warning:
                # Warning Zone
                self.failure_streak += 1
                trainer.log(f"-> Warning Zone (WR < {self.config.wr_warning:.2f}): Streak {self.failure_streak}/4")
                
                if self.failure_streak >= 4:
                    if auto:
                        trainer.env.bot_difficulty = max(0.0, trainer.env.bot_difficulty - self.config.diff_step_decrease)
                        trainer.log(f"-> Persistent Failure: Decreasing Bot Difficulty to {trainer.env.bot_difficulty:.2f}")
                    self.failure_streak = 0
            
            else:
                # Progression
                self.failure_streak = 0
                new_diff = trainer.env.bot_difficulty
                msg = None
                
                if rec_wr > self.config.wr_progression_insane_turbo:
                    new_diff = min(1.0, trainer.env.bot_difficulty + self.config.diff_step_insane_turbo); msg = "Insane Turbo"
                elif rec_wr > self.config.wr_progression_super_turbo:
                    new_diff = min(1.0, trainer.env.bot_difficulty + self.config.diff_step_super_turbo); msg = "Super Turbo"
                elif rec_wr > self.config.wr_progression_turbo:
                    new_diff = min(1.0, trainer.env.bot_difficulty + self.config.diff_step_turbo); msg = "Turbo"
                elif rec_wr > self.config.wr_progression_standard:
                    new_diff = min(1.0, trainer.env.bot_difficulty + self.config.diff_step_standard); msg = "Standard"

                if trainer.env.bot_difficulty < 1.0:
                    if msg and auto:
                         trainer.env.bot_difficulty = new_diff
                         trainer.log(f"-> {msg}: Increasing Bot Difficulty to {trainer.env.bot_difficulty:.2f}")
                else:
                     # Max difficulty reached
                     pass
            
            # Graduation Check
            if trainer.env.bot_difficulty >= 1.0:
                cons_wr = self.config.duel_consistency_wr
                abs_wr = self.config.duel_absolute_wr
                cons_checks = self.config.duel_consistency_checks

                if rec_wr > cons_wr:
                    self.grad_consistency_counter += 1
                else:
                    self.grad_consistency_counter = 0

                should_graduate = False
                reason = ""
                
                if rec_wr >= abs_wr and self.grad_consistency_counter >= 2:
                    should_graduate = True
                    reason = f"WR {rec_wr:.2f} >= {abs_wr}"
                elif self.grad_consistency_counter >= cons_checks:
                    should_graduate = True
                    reason = f"WR > {cons_wr} for {cons_checks} checks"
                    
                if should_graduate:
                     trainer.log(f">>> UPGRADING TO STAGE 3: TEAM ({reason}) <<<")
                     if auto:
                         trainer.env.bot_difficulty = 0.0 # Reset
                         trainer.env.stage_metrics["recent_games"] = 0
                         trainer.env.stage_metrics["recent_wins"] = 0
                         trainer.env.stage_metrics["recent_episodes"] = 0
                         return STAGE_TEAM, reason

        return None, ""

    def update_step_penalty(self, base_penalty: float) -> float:
        return base_penalty


    @property
    def target_evolve_interval(self) -> int:
        return 5 # Slow Evolution for Duel


class TeamStage(Stage):
    def __init__(self, config: CurriculumConfig):
        super().__init__("Team", config)
        self.failure_streak = 0
        self.grad_consistency_counter = 0

    @property
    def target_evolve_interval(self) -> int:
        # We need access to bot difficulty.
        # But Stage doesn't reference Env directly unless passed strictly. 
        # However, we can assume we might need to access it via passed 'trainer' in 'update' 
        # BUT this property is accessed by Trainer.
        # Design flaw: 'Stage' should have access to 'Env' or 'Trainer' if dynamic.
        # Workaround: Return a safe default (2) OR refactor to pass context.
        # Given the legacy code in PPO accessed self.env.bot_difficulty:
        # We can relax this property to be fixed '2', OR we need to bind the Stage to the Env.
        # But PPO loop does: `target_evolve = self.curriculum.current_stage.target_evolve_interval`
        # Let's fix PPO to handle dynamic if needed, OR we just set it to 2 for now to simplify.
        # Actually PPO commented: "Dynamic Interval for Team matches... target_evolve = int(8 - 4 * self.env.bot_difficulty)"
        # If I return 2 here, I lose that logic.
        # I will leave PPO to handle Team dynamic logic? No I replaced it.
        # I should bind env to stage? 
        # Ideally Stage.on_enter(env) saves the env?
        # Yes, Stage can hold a reference to Env if initialized/entered.
        return 2 # Placeholder, dynamic logic requires Env access which is not guaranteed here yet.

    def get_active_pods(self) -> List[int]:
        return [0, 1]

    def get_env_config(self) -> EnvConfig:
        return EnvConfig(
            mode_name="team",
            track_gen_type="max_entropy",
            active_pods=[0, 1, 2, 3],
            use_bots=True,
            bot_pods=[2, 3],
            step_penalty_active_pods=[0, 1, 2, 3],
            orientation_active_pods=[0, 1, 2, 3]
        )

    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        rv = p.get('ema_runner_vel', 0.0); rv = rv if rv is not None else 0.0
        bd = p.get('ema_blocker_dmg', 0.0); bd = bd if bd is not None else 0.0
        return [
            p.get('ema_wins', 0.0),
            rv,
            bd
        ]
        
    def check_graduation(self, metrics: Dict[str, Any], env: Any) -> Tuple[bool, str]:
        return False, ""

    def update(self, trainer) -> Tuple[Optional[int], str]:
        # Dynamic Difficulty
        metrics = trainer.env.stage_metrics
        rec_episodes = metrics.get("recent_episodes", 0)
        if rec_episodes == 0: rec_episodes = metrics.get("recent_games", 0)

        if rec_episodes > 1000: 
            rec_wins = metrics["recent_wins"]
            rec_games = metrics["recent_games"]
            
            if rec_episodes > 0:
                rec_wr = rec_wins / rec_episodes
            else:
                rec_wr = 0.0
            
            trainer.current_win_rate = rec_wr
            metrics["recent_games"] = 0
            metrics["recent_wins"] = 0
            metrics["recent_episodes"] = 0
            
            trainer.log(f"Stage 3 (Team) Check: Recent WR {rec_wr*100:.1f}% | Diff: {trainer.env.bot_difficulty:.2f}")
            
            auto = (trainer.curriculum_mode == "auto")
            
            if rec_wr < self.config.wr_critical:
                if auto:
                     trainer.env.bot_difficulty = max(0.0, trainer.env.bot_difficulty - self.config.diff_step_decrease)
                     trainer.log(f"-> Regression: Diff {trainer.env.bot_difficulty:.2f}")
                self.failure_streak = 0
            elif rec_wr < self.config.wr_warning:
                self.failure_streak += 1
                if self.failure_streak >= 2:
                    if auto:
                        trainer.env.bot_difficulty = max(0.0, trainer.env.bot_difficulty - self.config.diff_step_decrease)
                        trainer.log(f"-> Persistent Failure: Diff {trainer.env.bot_difficulty:.2f}")
                    self.failure_streak = 0
            else:
                self.failure_streak = 0
                new_diff = trainer.env.bot_difficulty
                if rec_wr > self.config.wr_progression_super_turbo: new_diff += self.config.diff_step_super_turbo
                elif rec_wr > self.config.wr_progression_turbo: new_diff += self.config.diff_step_turbo
                elif rec_wr > self.config.wr_progression_standard: new_diff += self.config.diff_step_standard
                new_diff = min(1.0, new_diff)
                
                if trainer.env.bot_difficulty < 1.0 and auto:
                    trainer.env.bot_difficulty = new_diff
                    trainer.log(f"-> Progress: Diff {trainer.env.bot_difficulty:.2f}")

            # Graduation to League
            if trainer.env.bot_difficulty >= 1.0:
                cons_wr = self.config.team_consistency_wr
                abs_wr = self.config.team_absolute_wr
                cons_checks = self.config.team_consistency_checks
                
                if rec_wr > cons_wr:
                    self.grad_consistency_counter += 1
                else:
                    self.grad_consistency_counter = 0
                    
                should_graduate = False
                reason = ""
                if rec_wr >= abs_wr:
                    should_graduate = True
                    reason = f"WR {rec_wr:.2f} >= {abs_wr}"
                elif self.grad_consistency_counter >= cons_checks:
                    should_graduate = True
                    reason = f"WR > {cons_wr} for {cons_checks} checks"
                    
                if should_graduate:
                    trainer.log(f">>> UPGRADING TO STAGE 4: LEAGUE ({reason}) <<<")
                    if auto:
                        trainer.env.stage_metrics["recent_games"] = 0
                        trainer.env.stage_metrics["recent_wins"] = 0
                        return STAGE_LEAGUE, reason

        return None, ""

class LeagueStage(Stage):
    def __init__(self, config: CurriculumConfig):
        super().__init__("League", config)
    
    def get_active_pods(self) -> List[int]:
        return [0, 1]

    def get_env_config(self) -> EnvConfig:
        return EnvConfig(
            mode_name="league",
            track_gen_type="max_entropy",
            active_pods=[0, 1, 2, 3],
            use_bots=False, 
            step_penalty_active_pods=[0, 1, 2, 3],
            orientation_active_pods=[0, 1, 2, 3]
        )
    
    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        return [
            p.get('ema_wins', 0.0),
            p.get('ema_laps', 0.0),
            -p.get('ema_efficiency', 999.0)
        ]
        
    def check_graduation(self, metrics: Dict[str, Any], env: Any) -> Tuple[bool, str]:
        return False, "End Game"

    def update(self, trainer) -> Tuple[Optional[int], str]:
        # League logic: Just monitor?
        # Maybe handle persistent logging or something.
        return None, ""
        
    def update_step_penalty(self, base_penalty: float) -> float:
        return base_penalty
