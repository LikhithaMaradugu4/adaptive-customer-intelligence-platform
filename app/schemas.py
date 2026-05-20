from typing import List
from pydantic import BaseModel


class IntentOutput(BaseModel):
    """
    Structured output schema
    for intent classification.
    """

    intent: List[str]
    confidence: float

class EmotionOutput(BaseModel):
    """
    Structured output schema
    for emotion detection.
    """

    emotion: str
    confidence: float 
    
class DecisionOutput(BaseModel):
    """
    Structured output schema
    for decision agent.
    """

    decision: str
    priority: str
    clarification_needed: bool
    clarification_question: str | None = None
    human_approval_required: bool
    approval_reason: str | None = None

class ResponseOutput(BaseModel):
    """
    Structured output schema
    for response generation.
    """

    response: str