from app.state import CustomerState
from app.graph import graph


state = CustomerState(
    query="Can I know about refund policy in the cpmpny just want to know?",
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

print(
    "\nRetrieved Docs:",
    result["retrieved_docs"]
)
print(
    "\nCustomer Profile:",
    result["customer_profile"]
)

print("\nDecision:", result["decision"])
print("Priority:", result["priority"])

print(
    "Clarification Needed:",
    result["clarification_needed"]
)

print(
    "Human Approval Required:",
    result["human_approval_required"]
)



print(
    "\nEscalation Details:",
    result.get("escalation_details")
)

print("\nFinal Response:\n")
print(result["response"])
