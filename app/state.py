from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid


class CustomerState(BaseModel):
    """
    Shared state passed across all agents in the LangGraph workflow.
    Each agent reads from and writes to this state.
    """
    customer_id: str = "CUST_001"
    
    # Original customer input
    query: str

    # Current conversation memory
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

    # Session tracking
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Language detection
    language: Optional[str] = None
    language_confidence: Optional[float] = None

    # Intent detection
    intent: Optional[List[str]] = None
    intent_confidence: Optional[float] = None

    # Emotion detection
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None

    # Memory agent output
    customer_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    # Customer profile agent output
    customer_profile: Optional[Dict[str, Any]] = None

    # RAG agent output
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    # Decision agent output
    decision: Optional[str] = None

    # Escalation tracking
    escalated: bool = False
    escalation_reason: Optional[str] = None

    # Response agent output
    response: Optional[str] = None

    # Error handling / supervisor tracking
    errors: List[str] = Field(default_factory=list)
    retry_count: int = 0

    # Execution tracing
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Observability / logging
    metadata: Dict[str, Any] = Field(default_factory=dict)# Decision system outputs
    decision: Optional[str] = None

    priority: str = "NORMAL"

    clarification_needed: bool = False

    clarification_question: Optional[str] = None

    human_approval_required: bool = False 
      
    approval_reason: Optional[str] = None


    # Escalation system
    escalated: bool = False
    
    escalation_type: Optional[str] = None
    
    escalation_reason: Optional[str] = None
    
    ticket_id: Optional[str] = None
    
    workflow_status: str = "ACTIVE"


    # Escalation system
    escalation_details: Optional[
        Dict[str, Any]
    ] = None

    response: Optional[str] = None

    requires_rag: bool = False

    tool_outputs: list = []