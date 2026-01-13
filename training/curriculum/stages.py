from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from training.curriculum.base import Stage
from config import TrainingConfig, CurriculumConfig, EnvConfig
from simulation.env import (
    RW_WIN, RW_LOSS, RW_CHECKPOINT, RW_CHECKPOINT_SCALE, 
    RW_PROGRESS, RW_MAGNET, RW_COLLISION_RUNNER, RW_COLLISION_BLOCKER, 
    RW_STEP_PENALTY, RW_ORIENTATION, RW_WRONG_WAY, RW_COLLISION_MATE,
    STAGE_NURSERY, STAGE_SOLO, STAGE_DUEL_FUSED, STAGE_RUNNER, STAGE_TEAM, STAGE_LEAGUE
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
                # trainer.reward_weights_tensor[:, RW_CHECKPOINT] = 2000.0
                # trainer.log("Config Update: Resetting RW_CHECKPOINT to 2000.0 for Stage 1+")
                # REMOVED: Respect Global Config (RW_CHECKPOINT=500, RW_LAP=2000)
                trainer.log("Config Update: Entering Unified Duel Stage (Difficulty Reset).")
                
                return STAGE_DUEL_FUSED, f"Eff {avg_eff:.1f} < {self.config.solo_efficiency_threshold}"
        
        return None, ""
        
class BlockerStage(Stage):
    def __init__(self, config: CurriculumConfig):
        super().__init__("BlockerAcademy", config)
        self.failure_streak = 0
        self.cons_streak = 0
        self.grad_consistency_counter = 0

    def get_active_pods(self) -> List[int]:
        # Only Pod 0 is the Learner. Pod 2 is the Bot.
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
        # Objectives: Denial Rate (Primary), Blocker Score (Secondary), Novelty
        # We want to maximize Denial Rate and Collision Impact.
        
        denials = p.get('ema_denials', 0.0)
        blocker_score = p.get('ema_blocker_score', 0.0) # Includes collisions + denials
        
        # Novelty only matters if agent is competent
        competence = denials
        nov = p.get('novelty_score', 0.0) * 100.0
        if competence < 0.05:
            nov = 0.0
            
        # Objectives: [DenialRate(Max), BlockerScore(Max), Novelty(Max)]
        return [
            denials,
            blocker_score,
            nov
        ]

    def update(self, trainer) -> Tuple[Optional[int], str]:
        # Dynamic Difficulty based on Denial Rate (DR)
        metrics = trainer.env.stage_metrics
        rec_episodes = metrics.get("recent_episodes", 0)
        if rec_episodes == 0:
             rec_episodes = metrics.get("recent_games", 0)

        if rec_episodes > 1000: 
            rec_games = metrics.get("recent_games", 0) # Finished games (Runner finished or timeout)
            
            # Denial Rate
            rec_denials = metrics.get("recent_denials", 0)
            rec_dr = rec_denials / rec_episodes if rec_episodes > 0 else 0.0
            
            # Blocker Hits (Collisions)
            rec_hits = metrics.get("blocker_collisions", 0)
            avg_hits = rec_hits / rec_games if rec_games > 0 else 0.0
            
            # Reset Recent
            metrics["recent_games"] = 0
            metrics["recent_wins"] = 0
            metrics["recent_denials"] = 0
            metrics["blocker_collisions"] = 0
            metrics["recent_episodes"] = 0
            
            trainer.log(f"Stage 2 (Blocker Academy) Check: DR {rec_dr*100:.1f}% | AvgHits {avg_hits:.1f} | Diff: {trainer.env.bot_difficulty:.2f}")
            
            auto = (trainer.curriculum_mode == "auto")
            
            # --- Difficulty Adjustment ---
            # User Request: Augment bot difficulty when DR > 70%
            if rec_dr > 0.70:
                # Progression
                self.failure_streak = 0
                # Use configurable max difficulty for progression
                max_diff = getattr(self.config, 'duel_graduation_difficulty', 1.0)
                
                if auto and trainer.env.bot_difficulty < max_diff:
                    # Increase difficulty
                    # Boost by standard step or turbo if really crushing it (>90%)
                    step = self.config.diff_step_turbo if rec_dr > 0.90 else self.config.diff_step_standard
                    trainer.env.bot_difficulty = min(max_diff, trainer.env.bot_difficulty + step)
                    trainer.log(f"-> Strong Defense (DR > 70%): Increasing Bot Difficulty to {trainer.env.bot_difficulty:.2f}")
                    
            elif rec_dr < 0.10:
                 # Regression / Safety Net
                 # If getting crushed (<10% Denial), possibly reduce difficulty?
                 self.failure_streak += 1
                 if self.failure_streak >= 5 and auto:
                      trainer.env.bot_difficulty = max(0.0, trainer.env.bot_difficulty - self.config.diff_step_decrease)
                      trainer.log(f"-> Weak Defense (DR < 10%): Decreasing Bot Difficulty to {trainer.env.bot_difficulty:.2f}")
                      self.failure_streak = 0
            else:
                 # Maintain
                 pass

            # --- Graduation Check ---
            # Criteria: High Denial Rate (> 80%) AND Sustained Pressure (Collision Steps)
            # Need to verify config names. Using generic hardcoded fallback if config missing.
            grad_dr = getattr(self.config, 'duel_graduation_denial_rate', 0.80) 
            grad_hits = getattr(self.config, 'duel_graduation_collision_steps', 60.0)
            
            # Note: We overwrite the config's "0.05" default with a higher standard here? 
            # Or reliance on config? 
            # Config currently has 0.05. I should probably use a higher value for "Blocker Academy".
            # Overriding config for now to enforce "Blocker Mastery".
            # grad_dr = 0.80 
            
            # Use configurable max difficulty
            grad_diff = getattr(self.config, 'duel_graduation_difficulty', 1.0)
            
            # Allow graduation if we are AT or ABOVE the target difficulty
            # (Use >= to handle float precision close calls or explicit caps)
            passed = (trainer.env.bot_difficulty >= grad_diff) and (rec_dr > grad_dr)
            
            if passed:
                 self.grad_consistency_counter += 1
            else:
                 self.grad_consistency_counter = 0
            
            checks = getattr(self.config, 'duel_graduation_checks', 5)
            
            if self.grad_consistency_counter >= checks:
                 reason = f"Blocker Mastery: DR {rec_dr:.2f} > {grad_dr} (Diff {trainer.env.bot_difficulty:.2f}) for {checks} checks"
                 trainer.log(f">>> GRADUATION FROM BLOCKER ACADEMY! {reason} <<<")
                 return STAGE_RUNNER, reason
            


        return None, ""


    def update_step_penalty(self, base_penalty: float) -> float:
        return base_penalty

    @property
    def target_evolve_interval(self) -> int:
        return 5 # Unified stage needs more time for pairing variations


class RunnerStage(Stage):
    def __init__(self, config: CurriculumConfig):
        super().__init__("RunnerAcademy", config)
        self.failure_streak = 0
        self.grad_consistency_counter = 0

    @property
    def target_evolve_interval(self) -> int:
        return 5 # Slower evolution to allow PPO convergence

    def get_active_pods(self) -> List[int]:
        # Agent (0), Bot(2)
        return [0]

    def get_env_config(self) -> EnvConfig:
        return EnvConfig(
            mode_name="duel",
            track_gen_type="max_entropy",
            active_pods=[0, 2],
            use_bots=True, 
            bot_pods=[2],
            step_penalty_active_pods=[0], # Penalty only for Runner
            orientation_active_pods=[0, 2]
        )

    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        # Objectives: Win Rate (Primary), Efficiency (Secondary), Novelty
        wins = p.get('ema_wins', 0.0)
        eff = p.get('ema_efficiency', 999.0)
        
        nov = p.get('novelty_score', 0.0) * 100.0
        
        return [
            wins,
            -eff,
            nov
        ]

    def update(self, trainer) -> Tuple[Optional[int], str]:
        # Dynamic Difficulty based on Win Rate (WR)
        metrics = trainer.env.stage_metrics
        rec_episodes = metrics.get("recent_episodes", 0)
        # Fallback to games if episodes not tracked
        if rec_episodes == 0:
            rec_episodes = metrics.get("recent_games", 0)
            
        rec_wins = metrics.get("recent_wins", 0)
        
        # Avoid division by zero
        rec_wr = rec_wins / max(1, rec_episodes)

        if rec_episodes > 1000: 
            rec_games = metrics.get("recent_games", 0)
            
            # Win Rate = Wins / Total Episodes
            rec_wins = metrics.get("recent_wins", 0)
            # rec_wins is total wins. 
            
            rec_wr = rec_wins / rec_episodes if rec_episodes > 0 else 0.0
            
            # Reset Recent
            metrics["recent_games"] = 0
            metrics["recent_wins"] = 0
            metrics["recent_episodes"] = 0
            
            trainer.log(f"Stage 3 (Runner Academy) Check: WR {rec_wr*100:.1f}% | Diff: {trainer.env.bot_difficulty:.2f}")
            
            auto = (trainer.curriculum_mode == "auto")
            
            # --- Difficulty Adjustment (Progression) ---
            # Increase difficulty if winning easily
            # Increase difficulty if winning easily
            max_diff = getattr(self.config, 'team_graduation_difficulty', 1.0)
            
            if trainer.env.bot_difficulty < max_diff:
                step = 0.0
                if rec_wr > self.config.wr_progression_insane_turbo: # 0.85
                    step = self.config.diff_step_insane_turbo # 0.05
                    desc = "INSANE TURBO"
                elif rec_wr > self.config.wr_progression_super_turbo: # 0.70
                    step = self.config.diff_step_super_turbo # 0.03
                    desc = "SUPER TURBO"
                elif rec_wr > self.config.wr_progression_turbo: # 0.60
                    step = self.config.diff_step_turbo # 0.02
                    desc = "TURBO"
                elif rec_wr > self.config.wr_progression_standard: # 0.55
                    step = self.config.diff_step_standard # 0.01
                    desc = "Standard"
                
                if step > 0.0 and auto:
                     trainer.env.bot_difficulty = min(max_diff, trainer.env.bot_difficulty + step)
                     trainer.log(f"-> Progression ({desc}): WR {rec_wr*100:.1f}% > Threshold. Increasing Difficulty to {trainer.env.bot_difficulty:.2f}")
            elif rec_wr < self.config.wr_critical and trainer.env.bot_difficulty > 0.0:
                 if auto:
                     step = self.config.diff_step_decrease
                     trainer.env.bot_difficulty = max(0.0, trainer.env.bot_difficulty - step)
                     trainer.log(f"-> Struggling (WR < {self.config.wr_critical*100:.0f}%): Decreasing Bot Difficulty to {trainer.env.bot_difficulty:.2f}")

            # --- Graduation Check ---
            # Criteria: High Win Rate against High Difficulty Blocker
            grad_wr = getattr(self.config, 'runner_graduation_win_rate', 0.80)
            grad_diff = getattr(self.config, 'team_graduation_difficulty', 1.0) 
            
            passed = (rec_wr > grad_wr) and (trainer.env.bot_difficulty >= grad_diff)
            
            if passed:
                 self.grad_consistency_counter += 1
                 
                 # Early Graduation for Elite Performance
                 if rec_wr >= 0.99:
                      reason = f"Elite Performance: WR {rec_wr:.1%} >= 99% (Diff {trainer.env.bot_difficulty:.2f})"
                      trainer.log(f">>> EARLY GRADUATION FROM RUNNER ACADEMY! {reason} <<<")
                      return STAGE_TEAM, reason
            else:
                 self.grad_consistency_counter = 0
                 
            checks = getattr(self.config, 'runner_graduation_checks', 5)
            
            if self.grad_consistency_counter >= checks:
                 reason = f"Competence: WR {rec_wr:.2f} > {grad_wr} (Diff {trainer.env.bot_difficulty:.2f}) for {checks} checks"
                 trainer.log(f">>> GRADUATION FROM RUNNER ACADEMY! {reason} <<<")
                 return STAGE_TEAM, reason

        return None, ""

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
        # I should bind env to stage? 
        # Ideally Stage.on_enter(env) saves the env?
        # Yes, Stage can hold a reference to Env if initialized/entered.
        # Team Mode requires more time for complex strategies (Blocking/Coop) to emerge
        # and for the agent to stabilize its coordination policy.
        # 2 Iterations is too fast (aggressive culling). Increasing to 5.
        return 5

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
        # Objectives: Win Rate, Spirit, Novelty
        eff = p.get('ema_efficiency', 999.0); eff = eff if eff is not None else 999.0
        return [
            p.get('ema_wins', 0.0),
            p.get('ema_denials', 0.0), # Carry over Blocker skills
            p.get('ema_blocker_dmg', 0.0), # Carry over Aggression
            p.get('team_spirit', 0.0),
            -eff
        ]

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
            
            trainer.log(f"Stage 4 (Team) Check: Recent WR {rec_wr*100:.1f}% | Diff: {trainer.env.bot_difficulty:.2f}")
            
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
                
                max_diff = self.config.team_graduation_difficulty
                if trainer.env.bot_difficulty < max_diff and auto:
                    if new_diff > max_diff: new_diff = max_diff
                    trainer.env.bot_difficulty = new_diff
                    trainer.log(f"-> Progress: Diff {trainer.env.bot_difficulty:.2f} (Capped at {max_diff})")

            # Graduation to League
            # [FIX]: Require Full Team Spirit (1.0) before considering graduation
            if trainer.team_spirit >= 1.0 and trainer.env.bot_difficulty >= self.config.team_graduation_difficulty:
                min_wr = self.config.team_graduation_win_rate
                checks = self.config.team_graduation_checks
                
                if rec_wr >= min_wr:
                    self.grad_consistency_counter += 1
                else:
                    # Reset consistency if spirit not ready (keep training)
                    # Use warning log occasionally?
                    # Determine progress of spirit annealing
                    self.grad_consistency_counter = 0
                
                if self.grad_consistency_counter >= checks:
                    reason = f"Competence: WR {rec_wr:.2f} >= {min_wr} & Spirit {trainer.team_spirit:.1f} (Diff {trainer.env.bot_difficulty:.2f})"
                    trainer.log(f">>> UPGRADING TO STAGE 5: LEAGUE ({reason}) <<<")
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

    def update(self, trainer) -> Tuple[Optional[int], str]:
        # League logic: Just monitor?
        # Maybe handle persistent logging or something.
        return None, ""
        
    def update_step_penalty(self, base_penalty: float) -> float:
        return base_penalty
