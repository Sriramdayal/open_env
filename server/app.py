from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action, Observation, State
from server.llm_env import LLMEnv

app = FastAPI(title="LLM Control OpenEnv")

# In-memory store for environments per episode id and overall states
envs = {}
completed_episodes = {}

class ResetRequest(BaseModel):
    task: str = "easy"

class StepRequest(BaseModel):
    action: Action
    episode_id: str | None = None

class GraderRequest(BaseModel):
    episode_id: str

# Default global environment to satisfy simple paths
default_env = LLMEnv()

@app.get("/", response_class=HTMLResponse)
async def serve_gui():
    path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "GUI index.html not found. Check the root directory."

@app.post("/reset", response_model=Observation)
async def reset(req: ResetRequest):
    if req.task not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid task")
    
    env = LLMEnv(task=req.task)
    obs = env.reset()
    envs[env.state.episode_id] = env
    
    # Also set default env to the latest reset for easy single-agent testing
    global default_env
    default_env = env
    
    return obs

@app.post("/step")
async def step(req: StepRequest):
    # Retrieve env
    env = default_env
    if req.episode_id and req.episode_id in envs:
        env = envs[req.episode_id]
        
    obs, reward, done, info = env.step(req.action)
    
    if done:
        # Save cumulative reward for grading
        completed_episodes[env.state.episode_id] = {
            "reward": env.state.cumulative_reward,
            "bounds": env._reward_bounds()
        }
        
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state", response_model=State)
async def get_state(episode_id: str | None = None):
    env = default_env
    if episode_id and episode_id in envs:
        env = envs[episode_id]
    return env.state

@app.post("/baseline")
async def run_baseline():
    import subprocess
    try:
        # baseline.py should be in the directory above server
        baseline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "baseline.py")
        result = subprocess.run([sys.executable, baseline_path], capture_output=True, text=True, check=True)
        # Parse the output to return the dict
        # We expect JSON or eval-able output from baseline, or simply look at the final prints
        # But this implies we should structure baseline.py to just run the tasks
        # Or we can just run the baseline logic directly here if we want API. 
        # For safety, let's just execute it and return the raw output or parse a standard format.
        
        # We'll just run our logic from baseline script here directly if the subprocess is too complex,
        # but the prompt says POST /baseline runs baseline.py, so we will return stdout.
        # Actually, let's try to extract JSON from the stdout.
        import json
        out = result.stdout.strip().splitlines()[-1] 
        # assume last line is valid JSON dict
        scores = json.loads(out)
        return scores
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline error: {str(e)}")

@app.post("/grader")
async def grader(req: GraderRequest):
    if req.episode_id not in completed_episodes:
        # Check active envs
        if req.episode_id in envs:
            env = envs[req.episode_id]
            r = env.state.cumulative_reward
            b_min, b_max = env._reward_bounds()
            norm = (r - b_min) / (b_max - b_min)
            return {"score": max(0.0, min(1.0, norm))}
            
        raise HTTPException(status_code=404, detail="Episode not found or not finished")
        
    data = completed_episodes[req.episode_id]
    r = data["reward"]
    b_min, b_max = data["bounds"]
    norm = (r - b_min) / (b_max - b_min)
    
    # Clip to [0, 1]
    norm = max(0.0, min(1.0, norm))
    return {"score": norm}

@app.get("/tasks")
async def get_tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "action_schema": Action.model_json_schema()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
