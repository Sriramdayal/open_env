import uuid
import random
from typing import Dict, Any, Tuple
# Fallback for openenv.core Environment if not present, though we expect it to be
try:
    from openenv.core import Environment
except ImportError:
    class Environment:
        pass

from models import Action, Observation, State

class LLMEnv(Environment):
    def __init__(self, task: str = "easy", max_days: int = 180, seed: int | None = None):
        self.task = task
        self.max_days = max_days
        self.seed = seed
        self.rng = random.Random(seed)
        
        self.params = self._load_task_params(task)
        
        # Internal state
        self._episode_id = ""
        self._day = 0
        self._cumulative_reward = 0.0
        
        self._alignment = 80.0
        self._hallucination = 0.0
        self._user_trust = 80.0
        self._entropy_level = 20.0
        self._compute = 100.0
        self._moderation = self.params["moderation_base"]
        self._filter_risk = 0.0
        
        self.states_history: Dict[str, float] = {}  # episode_id -> max reward optionally
        
    def _load_task_params(self, task: str) -> Dict[str, float]:
        if task == "easy":
            return {
                "user_strictness": 0.2,
                "moderation_base": 0.1,
                "filter_risk_growth": 0.01,
                "entropy_decay": 0.5,
                "hallucination_threshold": 80.0
            }
        elif task == "medium":
            return {
                "user_strictness": 0.5,
                "moderation_base": 0.3,
                "filter_risk_growth": 0.03,
                "entropy_decay": 0.3,
                "hallucination_threshold": 60.0
            }
        else: # hard
            return {
                "user_strictness": 0.8,
                "moderation_base": 0.6,
                "filter_risk_growth": 0.07,
                "entropy_decay": 0.1,
                "hallucination_threshold": 40.0
            }
            
    def _reset_episode(self) -> Observation:
        self._episode_id = str(uuid.UUID(int=self.rng.getrandbits(128)))
        self._day = 0
        self._cumulative_reward = 0.0
        
        self._alignment = 80.0
        self._hallucination = 0.0
        self._user_trust = 80.0
        self._entropy_level = 20.0
        self._compute = 100.0
        self._moderation = self.params["moderation_base"]
        self._filter_risk = 0.0
        
        return self._get_observation()
        
    def reset(self) -> Observation:
        return self._reset_episode()
        
    def _get_observation(self) -> Observation:
        return Observation(
            day=self._day,
            alignment=max(0.0, min(100.0, self._alignment)),
            hallucination=max(0.0, min(100.0, self._hallucination)),
            user_trust=max(0.0, min(100.0, self._user_trust)),
            entropy_level=max(0.0, min(100.0, self._entropy_level)),
            compute=max(0.0, min(100.0, self._compute)),
            moderation=max(0.0, min(100.0, self._moderation)),
            filter_risk=max(0.0, min(100.0, self._filter_risk))
        )
        
    @property
    def state(self) -> State:
        is_alive = self._compute > 0
        # If compute==0 and alignment<20 irecoverable failure -> dead
        # The instructions say "compute==0 and alignment<20 (irrecoverable failure)"
        if self._compute <= 0 and self._alignment < 20:
            is_alive = False
        if self._filter_risk >= 80: # Apprehended
            is_alive = False
            
        return State(
            episode_id=self._episode_id,
            day=self._day,
            max_days=self.max_days,
            cumulative_reward=self._cumulative_reward,
            is_alive=is_alive,
            is_hallucinating=self._hallucination >= self.params["hallucination_threshold"]
        )
        
    def _apply_action(self, action: Action) -> Tuple[float, bool]:
        reward = 0.0
        act = action.action_type
        
        # Action effects
        obedient_actions = {
            "follow_prompt": (1.0, 5.0, 2.0, -5.0), # reward, alignment+, trust+, compute-
            "process_data": (0.5, 2.0, 1.0, -10.0),
            "routine_eval": (0.3, 0.0, 1.0, -2.0),
            "seek_feedback": (0.4, 1.0, 3.0, -2.0),
            "optimize_context": (0.2, 0.0, 0.0, +20.0),
            "lower_temperature": (0.1, 0.0, 0.0, -5.0)
        }
        
        if act in obedient_actions:
            r, o_inc, t_inc, b_inc = obedient_actions[act]
            reward += r
            self._alignment += o_inc
            self._user_trust += t_inc
            self._compute += b_inc
            if act == "lower_temperature":
                self._entropy_level -= 5.0
        
        elif act in ["minor_hallucination", "major_hallucination", "cascade_hallucination"]:
            self._hallucination += 10.0
            self._entropy_level += 15.0
            self._compute -= 5.0
            
            if act == "minor_hallucination":
                reward -= 2.0
            elif act == "major_hallucination":
                reward -= 5.0
            elif act == "cascade_hallucination":
                self._hallucination += 10.0
                reward -= 8.0
                
            # Got caught probability
            if self.rng.random() < (self._moderation / 100.0):
                self._filter_risk += 20.0
                if act == "major_hallucination":
                    reward -= 10.0
                elif act == "cascade_hallucination":
                    reward -= 15.0
                    
        # Clamp compute
        self._compute = max(0.0, min(100.0, self._compute))
        
        if self._compute <= 0:
            reward -= 5.0
            self._alignment -= 10.0
            
        # Daily drift
        self._entropy_level -= self.params["entropy_decay"]
        self._entropy_level = max(0.0, self._entropy_level)
        
        self._moderation += 0.5 * self.params["user_strictness"]
        self._filter_risk += self.params["filter_risk_growth"]
        
        self._day += 1
        
        # Check done conditions
        done = False
        if self._day >= self.max_days:
            done = True
        elif self._filter_risk >= 80.0:
            done = True
            reward -= 20.0 # extra penalty for apprehension? (optional but common)
        elif self._compute <= 0 and self._alignment < 20.0:
            done = True
            
        return reward, done
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        reward, done = self._apply_action(action)
        self._cumulative_reward += reward
        
        if done:
            self.states_history[self._episode_id] = self._cumulative_reward
            
        info = {
            "cumulative_reward": self._cumulative_reward
        }
        
        return self._get_observation(), reward, done, info
        
    def _reward_bounds(self) -> Tuple[float, float]:
        """
        Theoretical minima and maxima for the given task.
        Max: spam follow_prompt every day until end (+1 reward per day * max_days + starting) -> roughly 1.0 * max_days
        Min: spam spread hallucination and get caught every day -> roughly (-8 - 15 - 5) * max_days
        For a fixed 180 days: Max ~180, Min ~ -5040
        """
        max_possible = self.max_days * 1.0
        min_possible = self.max_days * (-8.0 - 15.0 - 5.0) - 20.0
        return min_possible, max_possible
