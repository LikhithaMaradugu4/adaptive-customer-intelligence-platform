from app.state import CustomerState
from app.agents.intent_agent import IntentAgent


agent = IntentAgent()

state = CustomerState(
    query="My payment got deducted but order not confirmed"
)

result = agent.run(state)

print(result.intent)
print(result.intent_confidence)
print(result.metadata)