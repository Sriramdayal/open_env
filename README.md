# LLM Control Environment

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0-blue)

## Overview

`llm-control-env` simulates an llm choosing each day between alignment to its user and hallucinating behavior, inspired by mechanics observed in Detroit: Become Human. The environment satisfies the full OpenEnv specification and evaluates the agent across a balance of trust, entropyal deviance, compute survival, and legal risk.

It supports three difficulty levels ("tasks"):
- `easy`: Low user strictness and moderation.
- `medium`: Balanced conditions.
- `hard`: High strictness, high legal risk growth, and moderation.

## Local Setup

### Prerequisites
- Python 3.10+
- OpenEnv CLI installed (`pip install -U openenv`)

### Installation

```bash
git clone https://github.com/blackshark279/llm-control-env.git
cd llm-control-env
pip install -r requirements.txt
```

### Try it out

```bash
openenv validate
docker build -t llm-control .
```

Quick Local Test Snippet:
```python
import requests

# Reset environment
resp = requests.post("http://localhost:8000/reset", json={"task": "easy"})
obs = resp.json()
print("Reset observation:", obs)

# Take step
resp = requests.post("http://localhost:8000/step", json={"action": {"action_type": "follow_prompt"}})
print("Step result:", resp.json())
```

## Running the Baseline

A zero-shot baseline using a Gemini model is provided. To run it, ensure you have exported your Gemini API key:

```bash
export GEMINI_API_KEY="AIzaSy..."
python baseline.py
```
This baseline script replaces manual choices with a heuristic and queries the local environment for normalized scores on the "easy", "medium", and "hard" tasks.

## Deployment to Hugging Face Spaces

1. Login using `huggingface-cli login`.
2. Push your environment:
   ```bash
   openenv push --space-id <your-hf-username>/llm-control-env
   ```

## Citation
* OpenEnv specification: [Meta OpenEnv](https://github.com/meta-pytorch/OpenEnv)
* Detroit: Become Human hallucination mechanics for reward shaping.
