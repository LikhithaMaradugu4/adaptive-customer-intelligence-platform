from app.graph import route_after_decision
from app.state import CustomerState


def test_route_after_decision_clarify():
    state = CustomerState(query="?")
    state.decision = "CLARIFY"

    assert route_after_decision(state) == "response"


def test_route_after_decision_requires_rag():
    state = CustomerState(query="policy")
    state.decision = "RESPOND"
    state.requires_rag = True

    assert route_after_decision(state) == "rag"


def test_route_after_decision_escalation():
    state = CustomerState(query="payment failed")
    state.decision = "ESCALATE"

    assert route_after_decision(state) == "escalation"


def test_route_after_decision_default_response():
    state = CustomerState(query="general")
    state.decision = "RESPOND"
    state.requires_rag = False

    assert route_after_decision(state) == "response"
