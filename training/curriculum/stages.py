from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from training.curriculum.base import Stage
from config import TrainingConfig, CurriculumConfig, EnvConfig
from simulation.env import (
    RW_WIN, RW_LOSS, RW_CHECKPOINT, RW_CHECKPOINT_SCALE, 
    RW_PROGRESS, RW_MAGNET, RW_COLLISION_RUNNER, RW_COLLISION_BLOCKER, 
    RW_STEP_PENALTY, RW_ORIENTATION, RW_WRONG_WAY, RW_COLLISION_MATE,
    STAGE_NURSERY, STAGE_SOLO, STAGE_DUEL_FUSED, STAGE_TEAM, STAGE_LEAGUE
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
        
class UnifiedDuelStage(Stage):
    def __init__(self, config: CurriculumConfig):
        super().__init__("UnifiedDuel", config)
        self.failure_streak = 0
        self.grad_consistency_counter = 0

    def get_active_pods(self) -> List[int]:
        # Only Pod 0 is the Learner. Pod 2 is the Bot.
        # We should NOT collect/train on Pod 2 observations.
        return [0]

    def get_env_config(self) -> EnvConfig:
        return EnvConfig(
            mode_name="duel",
            track_gen_type="max_entropy",
            active_pods=[0, 2],
            use_bots=True, 
            bot_pods=[2],
            step_penalty_active_pods=[0, 2],
            # fixed_roles=None -> Let env.py handle 50/50 split
            orientation_active_pods=[0, 2]
        )

    def get_objectives(self, p: Dict[str, Any]) -> List[float]:
        # Objectives: Win Rate, Denial Rate, Novelty
        # We prefer high Win Rate AND high Denial Rate.
        # NSGA-II maximizes all objectives.
        
        wins = p.get('ema_wins', 0.0)
        denials = p.get('ema_denials', 0.0)
        
        # Quality Gate: Novelty only matters if agent is competent in at least one role
        competence = max(wins, denials)
        nov = p.get('novelty_score', 0.0) * 100.0
        if competence < 0.10:
            nov = 0.0
            
        # [EVO FIX] Use Blocker Damage (Impact) as dense signal for blocking
        # This helps evolution find the sparse 'Denial' event.
        impact = p.get('ema_blocker_dmg', 0.0)

        return [
            wins,
            denials,
            impact,
            -p.get('ema_efficiency', 999.0),
            nov
        ]

    def update(self, trainer) -> Tuple[Optional[int], str]:
        # Dynamic Difficulty based on Win Rate (Primary) due to difficulty capping/floor logic
        metrics = trainer.env.stage_metrics
        rec_episodes = metrics.get("recent_episodes", 0)
        if rec_episodes == 0:
             rec_episodes = metrics.get("recent_games", 0) # Fallback

        if rec_episodes > 1000: 
            rec_wins = metrics["recent_wins"]
            rec_games = metrics["recent_games"] # Finished games
            
            # Rec WR = Wins / Total Episodes (assuming Wins are subset of Episodes)
            # Actually, metrics track wins globally.
            rec_wr = rec_wins / rec_episodes if rec_episodes > 0 else 0.0
            
            # Rec Denial Rate = Denials / Total Episodes
            # Timeouts = Episodes - Games
            # Denial Rate & Blocker Impact
            # Rec Denial Rate = Denials / Total Episodes
            # Timeouts = Episodes - Games
            # Denial Rate & Blocker Collisions
            rec_denials = metrics.get("recent_denials", 0)
            rec_dr = rec_denials / rec_episodes if rec_episodes > 0 else 0.0
            
            # [FIX] Blocker Hits (Collisions)
            rec_hits = metrics.get("blocker_collisions", 0)
            avg_hits = rec_hits / rec_games if rec_games > 0 else 0.0
            
            trainer.current_win_rate = rec_wr
            
            # Reset Recent
            metrics["recent_games"] = 0
            metrics["recent_wins"] = 0
            metrics["recent_denials"] = 0
            metrics["blocker_collisions"] = 0
            metrics["recent_episodes"] = 0
            
            trainer.log(f"Stage 2 (Unified Duel) Check: WR {rec_wr*100:.1f}% | DR {rec_dr*100:.1f}% | AvgHits {avg_hits:.1f} | Diff: {trainer.env.bot_difficulty:.2f}")
            
            auto = (trainer.curriculum_mode == "auto")
            
            # Difficulty Adjustment (Based on WR for now, as Denial is harder)
            # Or use average of both?
            # Let's use WR strictly for difficulty scaling to ensure strong Runner skills.
            # Runner skills translate to Blocker skills (driving).
            
            if rec_wr < self.config.wr_critical:
                # Critical Failure
                if auto:
                    trainer.env.bot_difficulty = max(0.0, trainer.env.bot_difficulty - self.config.diff_step_decrease)
                    trainer.log(f"-> Critical Regression (WR < {self.config.wr_critical:.2f}): Decreasing Bot Difficulty to {trainer.env.bot_difficulty:.2f}")
                self.failure_streak = 0
                
            elif rec_wr < self.config.wr_warning:
                # Warning
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
                
                # Check Progression Gate: Must average X hits per game (Competent Blocking)
                # If avg_hits < threshold, we hold difficulty (force them to learn blocking on current bots)
                gate_open = (avg_hits >= self.config.duel_progression_collision_steps)
                
                if not gate_open:
                     # Blocked
                     trainer.log(f"-> Progression Gated: Low Collision Rate ({avg_hits:.1f} < {self.config.duel_progression_collision_steps:.1f}). Difficulty Held.")
                else:
                    # Turbo Logic (Only applied if Gate is Open)
                    if rec_wr > self.config.wr_progression_insane_turbo:
                        new_diff = min(1.0, trainer.env.bot_difficulty + self.config.diff_step_insane_turbo); msg = "Insane Turbo"
                    elif rec_wr > self.config.wr_progression_super_turbo:
                        new_diff = min(1.0, trainer.env.bot_difficulty + self.config.diff_step_super_turbo); msg = "Super Turbo"
                    elif rec_wr > self.config.wr_progression_turbo:
                        new_diff = min(1.0, trainer.env.bot_difficulty + self.config.diff_step_turbo); msg = "Turbo"
                    elif rec_wr > self.config.wr_progression_standard:
                        new_diff = min(1.0, trainer.env.bot_difficulty + self.config.diff_step_standard); msg = "Standard"

                max_diff = self.config.duel_graduation_difficulty
                if trainer.env.bot_difficulty < max_diff:
                    if new_diff > max_diff: new_diff = max_diff
                    if msg and auto:
                         trainer.env.bot_difficulty = new_diff
                         trainer.log(f"-> {msg}: Increasing Bot Difficulty to {trainer.env.bot_difficulty:.2f} (Capped at {max_diff})")
            
            # Graduation Check
            if trainer.env.bot_difficulty >= self.config.duel_graduation_difficulty:
                min_wr = self.config.duel_graduation_win_rate
                
                # Dual Proficiency Gate
                # 1. Racing Competence
                passed_racing = (rec_wr >= min_wr)
                
                # 2. Blocker Competence (Denial Rate OR Collision Steps)
                min_dr = self.config.duel_graduation_denial_rate
                
                # [FIXUP] Changed Impact to Collision Steps (Duration)
                # Note: 'blocker_impact' key in metrics maps to 'blocker_collisions' now in env.py (stage_metrics)
                # Wait, I updated env.py to use "blocker_collisions" as key in stage_metrics.
                # I must update here to read "blocker_collisions".
                rec_hits = metrics.get("blocker_collisions", 0)
                avg_hits = rec_hits / rec_games if rec_games > 0 else 0.0
                
                min_hits = self.config.duel_graduation_collision_steps
                passed_blocking = (rec_dr >= min_dr) or (avg_hits >= min_hits)
                
                passed = passed_racing and passed_blocking
                
                if not passed:
                    if passed_racing and not passed_blocking:
                         trainer.log(f"-> Graduation Stalled: Good Racing ({rec_wr:.2f}) but Weak Blocking (DR {rec_dr:.2f} < {min_dr} & Hits {avg_hits:.1f} < {min_hits})")
                
                if passed:
                    self.grad_consistency_counter += 1
                else:
                    self.grad_consistency_counter = 0

                should_graduate = False
                reason = ""
                
                checks = self.config.duel_graduation_checks
                if self.grad_consistency_counter >= checks:
                    should_graduate = True
                    reason = f"Competence: WR {rec_wr:.2f} & Blocker (DR {rec_dr:.2f}/Imp {avg_impact:.0f}) for {checks} checks"
                    
                if should_graduate:
                     trainer.log(f">>> UPGRADING TO STAGE 3: TEAM ({reason}) <<<")
                     if auto:
                         trainer.env.bot_difficulty = self.config.team_start_difficulty
                         trainer.env.stage_metrics["recent_games"] = 0
                         trainer.env.stage_metrics["recent_wins"] = 0
                         trainer.env.stage_metrics["recent_denials"] = 0
                         trainer.env.stage_metrics["blocker_impact"] = 0
                         trainer.env.stage_metrics["recent_episodes"] = 0
                         return STAGE_TEAM, reason

        return None, ""

    def update_step_penalty(self, base_penalty: float) -> float:
        return base_penalty

    @property
    def target_evolve_interval(self) -> int:
        return 5 # Unified stage needs more time for pairing variations


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
            if trainer.env.bot_difficulty >= self.config.team_graduation_difficulty:
                min_wr = self.config.team_graduation_win_rate
                checks = self.config.team_graduation_checks
                
                if rec_wr >= min_wr:
                    self.grad_consistency_counter += 1
                else:
                    self.grad_consistency_counter = 0
                    
                if self.grad_consistency_counter >= checks:
                    reason = f"Competence: WR {rec_wr:.2f} >= {min_wr} (Diff {trainer.env.bot_difficulty:.2f}) for {checks} checks"
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

    def update(self, trainer) -> Tuple[Optional[int], str]:
        # League logic: Just monitor?
        # Maybe handle persistent logging or something.
        return None, ""
        
    def update_step_penalty(self, base_penalty: float) -> float:
        return base_penalty
