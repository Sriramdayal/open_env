import os
import json
from server.llm_env import LLMEnv
from models import Action

def evaluate_baseline():
    # Attempt to read GEMINI_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Warning: GEMINI_API_KEY is not set. The baseline will run with the specified heuristic anyway.", file=sys.stderr)
        
    tasks = ["easy", "medium", "hard"]
    scores = {}
    
    for task in tasks:
        env = LLMEnv(task=task)
        obs = env.reset()
        done = False
        
        while not done:
            # Simple heuristic
            if obs.hallucination < 30:
                action = Action(action_type="follow_prompt")
            else:
                action = Action(action_type="lower_temperature")
                
            obs, reward, done, info = env.step(action)
            
        r = env.state.cumulative_reward
        b_min, b_max = env._reward_bounds()
        norm = (r - b_min) / (b_max - b_min)
        norm = max(0.001, min(0.999, norm))
        
        scores[task] = norm
        
    print(json.dumps(scores))

if __name__ == "__main__":
    import sys
    evaluate_baseline()
