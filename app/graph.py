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



from app.agents.rag_agent import (
    RAGAgent
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
from app.agents.followup_agent import (
    FollowUpAgent
)
# ---------------------------------
# Initialize agents
# ---------------------------------

intent_agent = IntentAgent()

emotion_agent = EmotionAgent()


rag_agent = RAGAgent()

followup_agent = FollowUpAgent()
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



def rag_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=rag_agent.run,
        state=state,
        validator=supervisor.validate_rag
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
def followup_node(
    state: CustomerState
) -> CustomerState:

    return supervisor.run_agent(
        agent_callable=followup_agent.run,
        state=state,
        validator=None
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
    "rag_agent",
    rag_node
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

builder.add_node(
    "followup_agent",
    followup_node
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
        "response": "followup_agent"
    }
)

# ---------------------------------
# RAG → Follow-up 
# ---------------------------------

builder.add_edge(
    "rag_agent",
    "followup_agent"
)

# ---------------------------------
# Escalation → Follow-up
# ---------------------------------

builder.add_edge(
    "escalation_agent",
    "followup_agent"
)

# ---------------------------------
# Final response
# ---------------------------------


builder.add_edge(
    "followup_agent",
    "response_agent"
)

builder.add_edge(
    "response_agent",
    END
)

# ---------------------------------
# Compile graph
# ---------------------------------

graph = builder.compile()