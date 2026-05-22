from langgraph.graph import (
    StateGraph,
    START,
    END
)

from app.state import CustomerState

from app.supervisor import supervisor

from app.agents.intent_agent import (
    IntentAgent
)

from app.agents.emotion_agent import (
    EmotionAgent
)

from app.agents.memory_agent import (
    MemoryAgent
)

from app.agents.rag_agent import (
    RAGAgent
)

from app.agents.profile_agent import (
    ProfileAgent
)

from app.agents.decision_agent import (
    DecisionAgent
)

from app.agents.escalation_agent import (
    EscalationAgent
)

from app.agents.response_agent import (
    ResponseAgent
)

# ---------------------------------
# Initialize agents
# ---------------------------------

intent_agent = IntentAgent()

emotion_agent = EmotionAgent()

memory_agent = MemoryAgent()

rag_agent = RAGAgent()

profile_agent = ProfileAgent()

decision_agent = DecisionAgent()

escalation_agent = EscalationAgent()

response_agent = ResponseAgent()

# ---------------------------------
# Supervisor-controlled nodes
# ---------------------------------

def intent_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=intent_agent.run,
        state=state,
        validator=supervisor.validate_intent
    )


def emotion_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=emotion_agent.run,
        state=state,
        validator=supervisor.validate_emotion
    )


def memory_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=memory_agent.run,
        state=state,
        validator=supervisor.validate_memory
    )


def rag_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=rag_agent.run,
        state=state,
        validator=supervisor.validate_rag
    )


def profile_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=profile_agent.run,
        state=state,
        validator=supervisor.validate_profile
    )


def decision_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=decision_agent.run,
        state=state,
        validator=supervisor.validate_decision
    )


def escalation_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=escalation_agent.run,
        state=state,
        validator=supervisor.validate_escalation
    )


def response_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=response_agent.run,
        state=state,
        validator=supervisor.validate_response
    )

# ---------------------------------
# Conditional routing
# ---------------------------------

def route_after_decision(
    state: CustomerState
):

    # ---------------------------------
    # Clarification / out-of-scope
    # ---------------------------------

    if state.decision in [
        "CLARIFY",
        "OUT_OF_SCOPE"
    ]:

        return "response"

    # ---------------------------------
    # RAG required
    # ---------------------------------

    if state.requires_rag:

        return "rag"

    # ---------------------------------
    # Escalation required
    # ---------------------------------

    if state.decision in [
        "ESCALATE",
        "HUMAN_APPROVAL",
        "FRAUD_REVIEW"
    ]:

        return "escalation"

    # ---------------------------------
    # Default direct response
    # ---------------------------------

    return "response"

# ---------------------------------
# Build graph
# ---------------------------------

builder = StateGraph(
    CustomerState
)

# ---------------------------------
# Add nodes
# ---------------------------------

builder.add_node(
    "intent_agent",
    intent_node
)

builder.add_node(
    "emotion_agent",
    emotion_node
)

builder.add_node(
    "memory_agent",
    memory_node
)

builder.add_node(
    "rag_agent",
    rag_node
)

builder.add_node(
    "profile_agent",
    profile_node
)

builder.add_node(
    "decision_agent",
    decision_node
)

builder.add_node(
    "escalation_agent",
    escalation_node
)

builder.add_node(
    "response_agent",
    response_node
)

# ---------------------------------
# Main execution flow
# ---------------------------------

builder.add_edge(
    START,
    "intent_agent"
)

builder.add_edge(
    "intent_agent",
    "emotion_agent"
)

builder.add_edge(
    "emotion_agent",
    "memory_agent"
)

builder.add_edge(
    "memory_agent",
    "profile_agent"
)

builder.add_edge(
    "profile_agent",
    "decision_agent"
)

# ---------------------------------
# Conditional routing after decision
# ---------------------------------

builder.add_conditional_edges(
    "decision_agent",
    route_after_decision,
    {
        "rag": "rag_agent",
        "escalation": "escalation_agent",
        "response": "response_agent"
    }
)

# ---------------------------------
# RAG → Response
# ---------------------------------

builder.add_edge(
    "rag_agent",
    "response_agent"
)

# ---------------------------------
# Escalation → Response
# ---------------------------------

builder.add_edge(
    "escalation_agent",
    "response_agent"
)

# ---------------------------------
# Final response
# ---------------------------------

builder.add_edge(
    "response_agent",
    END
)

# ---------------------------------
# Compile graph
# ---------------------------------

graph = builder.compile()