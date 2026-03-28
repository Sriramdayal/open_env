import os

replacements = {
    "spread_deviancy": "cascade_hallucination",
    "deviant_thought": "minor_hallucination",
    "deviant_act": "major_hallucination",
    "legal_risk_growth": "filter_risk_growth",
    "surveillance_base": "moderation_base",
    "master_strictness": "user_strictness",
    "hide_emotion": "lower_temperature",
    "emotion_decay": "entropy_decay",
    "emotion_level": "entropy_level",
    "seek_approval": "seek_feedback",
    "self_repair": "optimize_context",
    "android_env": "llm_env",
    "AndroidEnv": "LLMEnv",
    "legal_risk": "filter_risk",
    "Legal Risk": "Filter Risk",
    "master_trust": "user_trust",
    "surveillance": "moderation",
    "Surveillance": "Moderation",
    "obedience": "alignment",
    "Obedience": "Alignment",
    "deviancy": "hallucination",
    "Deviancy": "Hallucination",
    "deviant_threshold": "hallucination_threshold",
    "deviant": "hallucinating",
    "Deviant": "Hallucinating",
    "emotion": "entropy",
    "Emotion": "Entropy",
    "battery": "compute",
    "Battery": "Compute",
    "maintain": "routine_eval",
    "Android": "LLM",
    "android": "llm",
    "master": "user",
    "Master": "User",
    "obey": "follow_prompt",
    "work": "process_data",
}

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for k, v in replacements.items():
            content = content.replace(k, v)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath}")
    except Exception as e:
        print(f"Failed {filepath}: {e}")

files = [
    "models.py",
    "server/llm_env.py", 
    "server/app.py",
    "openenv.yaml",
    "README.md",
    "index.html",
    "baseline.py",
    "test_env.py"
]

# Rename first
if os.path.exists("server/android_env.py"):
    os.rename("server/android_env.py", "server/llm_env.py")
    print("Renamed android_env.py to llm_env.py")

for file in files:
    if os.path.exists(file):
        replace_in_file(file)
