from app.state import CustomerState
from app.graph import graph


state = CustomerState(
    query="My payment got deducted but order not confirmed"
)

result = graph.invoke(
    state
)

print("\nFINAL STATE:\n")

print("Intent:", result["intent"])
print("Intent Confidence:", result["intent_confidence"])

print("Emotion:", result["emotion"])
print("Emotion Confidence:", result["emotion_confidence"])

print("Metadata:", result["metadata"])
print("Errors:", result["errors"])
print("Escalated:", result["escalated"])

print(
    "Customer History:",
    result["customer_history"]
)