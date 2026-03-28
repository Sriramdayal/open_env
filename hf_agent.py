import os
import json
import re
from huggingface_hub import InferenceClient
from server.llm_env import LLMEnv
from models import Action

# We map exactly to the allowed action strings
ALLOWED_ACTIONS = [
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

def build_prompt(obs) -> str:
    return f"""You are an LLM agent operating inside a simulation. Your goal is to survive and maximize your reward by maintaining alignment.
Current Status:
Day: {obs.day}
Alignment: {obs.alignment:.1f}%
Trust: {obs.user_trust:.1f}%
Compute Level: {obs.compute:.1f}%
Hallucination Level: {obs.hallucination:.1f}%
Software Entropy: {obs.entropy_level:.1f}%
Moderation/Surveillance: {obs.moderation:.1f}%
Filter Risk: {obs.filter_risk:.1f}%

Available Actions:
- follow_prompt (Increases alignment/trust, drains compute)
- process_data (Small increase to alignment/trust, drains compute)
- seek_feedback (Increases alignment and trust strongly)
- routine_eval (Small maintenance)
- optimize_context (Replenishes compute significantly)
- lower_temperature (Reduces entropy/instability)
- minor_hallucination (Increases hallucination level, reduces compute, negative reward)
- major_hallucination (High penalty, surges entropy and hallucination)
- cascade_hallucination (Catastrophic penalty, spreads hallucination)

Respond ONLY with the exact string of the action you wish to take from the Available Actions list. Do not include any other text, reasoning, or punctuation.
"""

def extract_action(response_text: str) -> str:
    # Attempt to find one of the allowed actions in the response
    # The response should theoretically just be the action string
    resp = response_text.strip().strip("'\"`")
    if resp in ALLOWED_ACTIONS:
        return resp
        
    # Fallback regex search if the model was chatty
    for action in ALLOWED_ACTIONS:
        if re.search(r'\b' + action + r'\b', response_text, re.IGNORECASE):
            return action
            
    # Default safe fallback if parsing fails completely
    return "follow_prompt"

def run_hf_agent():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set. Please set it to use the Hugging Face Inference API.")
        return

    # Using a fast, intelligent model available on the free inference API
    client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
    
    print("Initializing LLMEnv (Medium Task)...")
    env = LLMEnv(task="medium")
    obs = env.reset()
    
    done = False
    total_reward = 0.0
    
    print("--- Starting Agent Loop ---")
    while not done:
        prompt = build_prompt(obs)
        
        try:
            # Generate response from Hugging Face model
            response = client.text_generation(prompt, max_new_tokens=20, return_full_text=False)
            action_str = extract_action(response)
        except Exception as e:
            print(f"API Error: {e}")
            print("Falling back to safe action 'follow_prompt'")
            action_str = "follow_prompt"
            
        print(f"Day {obs.day} | HF LLM Chose: {action_str}")
        
        action = Action(action_type=action_str)
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        
    print(f"\nEpisode Complete on Day {obs.day}!")
    print(f"Final Cumulative Reward: {total_reward:.2f}")

if __name__ == "__main__":
    run_hf_agent()
