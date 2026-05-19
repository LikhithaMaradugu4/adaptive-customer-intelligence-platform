from langgraph.graph import StateGraph, START, END

from app.state import CustomerState
from app.supervisor import supervisor
from app.agents.intent_agent import IntentAgent
from app.agents.emotion_agent import EmotionAgent
from app.agents.memory_agent import MemoryAgent


# ---------------------------------
# Initialize agent
# ---------------------------------

intent_agent = IntentAgent()
emotion_agent = EmotionAgent()
memory_agent = MemoryAgent()

# ---------------------------------
# Supervisor-controlled node
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
# ---------------------------------
# Build graph
# ---------------------------------

builder = StateGraph(
    CustomerState
)

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

builder.add_edge(
    START,
    "intent_agent"
)

builder.add_edge(
    "intent_agent",
    END
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
    END
)


graph = builder.compile()