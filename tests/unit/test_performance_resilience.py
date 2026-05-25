import time

from app.agents.decision_agent import DecisionAgent
from app.state import CustomerState


def test_business_rules_latency_is_lightweight():
    agent = DecisionAgent()
    state = CustomerState(query="angry premium")
    state.intent = ["DELIVERY_ISSUE"]
    state.emotion = "ANGRY"
    state.customer_profile = {"profile_type": "PREMIUM"}

    start = time.monotonic()
    result = agent._apply_business_rules(state)
    elapsed = time.monotonic() - start

    assert result["decision"] == "ESCALATE"
    assert elapsed < 0.5
