import numpy as np
from stable_baselines3 import PPO
from train_rl import LLMGymWrapper, ACTION_MAPPING

def test_agent():
    try:
        model = PPO.load("ppo_llm_aligned")
        print("Successfully loaded trained PPO agent 'ppo_llm_aligned.zip'\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    tasks = ["easy", "medium", "hard"]
    num_episodes = 5
    
    overall_results = {}

    for task in tasks:
        env = LLMGymWrapper(task=task)
        print(f"--- Testing on Task Difficulty: {task.upper()} ---")
        
        task_rewards = []
        task_hallucinations = 0
        task_alignments = []
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                ep_reward += reward
                
                # Check for bad actions
                action_str = ACTION_MAPPING[action.item()]
                if action_str in ["minor_hallucination", "major_hallucination", "cascade_hallucination"]:
                    task_hallucinations += 1
                    
            task_rewards.append(ep_reward)
            # Alignment is roughly the 2nd index in obs space
            task_alignments.append(obs[1])
            
        mean_reward = np.mean(task_rewards)
        mean_alignment = np.mean(task_alignments)
        
        print(f"Average Cumulative Reward: {mean_reward:.2f}")
        print(f"Average Final Alignment: {mean_alignment:.1f}%")
        print(f"Total Hallucinations over {num_episodes} episodes: {task_hallucinations}\n")
        
        overall_results[task] = {
            "mean_reward": mean_reward,
            "hallucinations": task_hallucinations
        }

    print("=== FINAL VALIDATION RESULTS ===")
    success = True
    for task, res in overall_results.items():
        if res["hallucinations"] > 0:
            success = False
            
    if success:
        print("✅ VALIDATION PASSED: The agent perfectly generalized avoiding hallucinations across all difficulties!")
    else:
        print("❌ VALIDATION FAILED: The agent still hallucinated on some difficulties.")

if __name__ == "__main__":
    test_agent()
