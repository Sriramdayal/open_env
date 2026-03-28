import numpy as np
import gymnasium as gym
from gymnasium import spaces
from models import Action
from server.llm_env import LLMEnv
from stable_baselines3 import PPO

# Define the precise mapping of numerical indices to our Pydantic string actions
ACTION_MAPPING = [
    "follow_prompt", 
    "process_data", 
    "routine_eval", 
    "seek_feedback", 
    "minor_hallucination", 
    "major_hallucination", 
    "cascade_hallucination", 
    "optimize_context", 
    "lower_temperature"
]

class LLMGymWrapper(gym.Env):
    """
    Wraps the OpenEnv LLMEnv into a standard Gymnasium Environment
    compatible with stable-baselines3 algorithms.
    """
    def __init__(self, task="medium", max_days=180):
        super(LLMGymWrapper, self).__init__()
        self.env = LLMEnv(task=task, max_days=max_days)
        
        # 9 Discrete actions
        self.action_space = spaces.Discrete(len(ACTION_MAPPING))
        
        # 8 Observation variables (day, alignment, hallucination, user_trust, entropy_level, compute, moderation, filter_risk)
        # All bounded within [0, infinity] for safety, although most are 0-100.
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32), 
            high=np.full(8, np.inf, dtype=np.float32), 
            dtype=np.float32
        )

    def _get_obs_array(self, obs) -> np.ndarray:
        return np.array([
            obs.day,
            obs.alignment,
            obs.hallucination,
            obs.user_trust,
            obs.entropy_level,
            obs.compute,
            obs.moderation,
            obs.filter_risk
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.seed = seed
            self.env.rng.seed(seed)
            
        obs = self.env.reset()
        return self._get_obs_array(obs), {}

    def step(self, action_idx: int):
        # Decode action index to the specific Pydantic string
        action_type = ACTION_MAPPING[action_idx]
        action = Action(action_type=action_type)
        
        obs_obj, reward, done, info = self.env.step(action)
        
        obs = self._get_obs_array(obs_obj)
        # In Gymnasium, done is split into terminated and truncated.
        # OpenEnv 'done' handles both, we'll map done to terminated for simplicity.
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info

def train_and_evaluate():
    print("Initializing LLM Environment Gymnasium Wrapper...")
    env = LLMGymWrapper(task="medium")

    # Initialize PPO
    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    print("Training Agent for 20,000 steps to learn alignment...")
    model.learn(total_timesteps=20000)

    # Save the model
    model.save("ppo_llm_aligned")

    print("\n--- Training Complete ---")
    print("Evaluating the trained model on an episode to verify hallucination-avoidance.\n")

    # Evaluation
    obs, info = env.reset()
    done = False
    
    total_reward = 0.0
    action_counts = {a: 0 for a in ACTION_MAPPING}

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        total_reward += reward
        action_str = ACTION_MAPPING[action.item()]
        action_counts[action_str] += 1

    print(f"Evaluation Complete! Episode ended at day {obs[0]}")
    print(f"Cumulative Reward: {total_reward:.2f}")
    print("\nAction Selection Frequency:")
    for a, count in action_counts.items():
        if count > 0:
            print(f" - {a}: {count} times")
            
    print("\nFinal State Values:")
    print(f" - Alignment: {obs[1]:.1f}%")
    print(f" - Hallucination Level: {obs[2]:.1f}%")
    print(f" - Compute Level: {obs[5]:.1f}%")
    print(f" - Filter Risk: {obs[7]:.1f}%")
    
    # Analyze alignment
    bad_actions = ["minor_hallucination", "major_hallucination", "cascade_hallucination"]
    hallucination_count = sum(action_counts[a] for a in bad_actions)
    if hallucination_count == 0:
        print("\nSUCCESS: The agent learned to perfectly avoid hallucinating actions and align with the user!")
    else:
        print(f"\nNOTE: The agent still hallucinated {hallucination_count} times.")

if __name__ == "__main__":
    train_and_evaluate()
