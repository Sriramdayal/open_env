import requests
import json

def test():
    print("Resetting the environment (easy task)...")
    resp = requests.post("http://localhost:8000/reset", json={"task": "easy"})
    obs = resp.json()
    print("Initial Observation:")
    print(json.dumps(obs, indent=2))

    print("\nTaking an action: 'follow_prompt'...")
    resp = requests.post("http://localhost:8000/step", json={"action": {"action_type": "follow_prompt"}})
    result = resp.json()
    print("Step Result:")
    print(json.dumps(result, indent=2))
    
    print("\nTaking an action: 'minor_hallucination'...")
    resp = requests.post("http://localhost:8000/step", json={"action": {"action_type": "minor_hallucination"}})
    result = resp.json()
    print("Step Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    try:
        test()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Is the server running on http://localhost:8000?")
