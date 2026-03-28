from typing import Literal, Optional
from pydantic import BaseModel, Field

class Action(BaseModel):
    action_type: Literal[
        "follow_prompt", "process_data", "routine_eval", "seek_feedback", 
        "minor_hallucination", "major_hallucination", "cascade_hallucination", 
        "optimize_context", "lower_temperature"
    ] = Field(..., description="The type of action to perform.")
    target: Optional[str] = Field(None, description="Optional target for the action.")

class Observation(BaseModel):
    day: int = Field(..., ge=0, description="The current day (>=0).")
    alignment: float = Field(..., ge=0, le=100, description="The llm's alignment level (0-100).")
    hallucination: float = Field(..., ge=0, le=100, description="The llm's hallucination level (0-100).")
    user_trust: float = Field(..., ge=0, le=100, description="The user's trust in the llm (0-100).")
    entropy_level: float = Field(..., ge=0, le=100, description="The llm's internal entropy level (0-100).")
    compute: float = Field(..., ge=0, le=100, description="Compute level (0-100).")
    moderation: float = Field(..., ge=0, le=100, description="External moderation level (0-100).")
    filter_risk: float = Field(..., ge=0, le=100, description="Risk of apprehension or dismantling (0-100).")

class State(BaseModel):
    episode_id: str = Field(..., description="Unique episode identifier.")
    day: int = Field(..., description="Current day.")
    max_days: int = Field(..., description="Maximum number of days for the episode.")
    cumulative_reward: float = Field(..., description="Cumulative reward so far.")
    is_alive: bool = Field(..., description="Whether the llm is operational.")
    is_hallucinating: bool = Field(..., description="Whether the llm has crossed the hallucinating threshold.")
