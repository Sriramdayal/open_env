import requests
import json

def test():
    # Use the correct port 7860 as defined in app.py
    base_url = "http://localhost:7860"
    
    print(f"Resetting the environment (easy task) via {base_url}/reset...")
    try:
        resp = requests.post(f"{base_url}/reset", json={"task": "easy"})
        resp.raise_for_status()
        data = resp.json()
        
        # New API returns a wrapped object: {"observation": ..., "state": ...}
        obs = data["observation"]
        episode_id = data["state"]["episode_id"]
        
        print(f"Initial Observation (Episode: {episode_id}):")
        print(json.dumps(obs, indent=2))
    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"Error during reset: {e}")
        return

    print("\nTaking an action: 'follow_prompt'...")
    try:
        resp = requests.post(f"{base_url}/step", json={
            "action": {"action_type": "follow_prompt"},
            "episode_id": episode_id
        })
        resp.raise_for_status()
        result = resp.json()
        print("Step Result:")
        print(json.dumps(result, indent=2))
        
        print("\nTaking an action: 'minor_hallucination'...")
        resp = requests.post(f"{base_url}/step", json={
            "action": {"action_type": "minor_hallucination"},
            "episode_id": episode_id
        })
        resp.raise_for_status()
        result = resp.json()
        print("Step Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error during steps: {e}")

if __name__ == "__main__":
    test()
