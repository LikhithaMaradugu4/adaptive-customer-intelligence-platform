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