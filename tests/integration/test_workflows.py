from app.graph import graph
from app.state import CustomerState
import app.graph as graph_module


class StubAgent:
    def __init__(self, run_func):
        self._run_func = run_func

    def run(self, state):
        return self._run_func(state)


def test_workflow_payment_issue(monkeypatch):
    def intent_run(state):
        state.intent = ["PAYMENT_ISSUE"]
        state.intent_confidence = 0.9
        return state

    def emotion_run(state):
        state.emotion = "NEUTRAL"
        state.emotion_confidence = 0.9
        return state

    def decision_run(state):
        state.decision = "RESPOND"
        state.requires_rag = False
        return state

    def response_run(state):
        state.response = "Payment response"
        return state

    monkeypatch.setattr(graph_module, "intent_agent", StubAgent(intent_run))
    monkeypatch.setattr(graph_module, "emotion_agent", StubAgent(emotion_run))
    monkeypatch.setattr(graph_module, "decision_agent", StubAgent(decision_run))
    monkeypatch.setattr(graph_module, "response_agent", StubAgent(response_run))

    state = CustomerState(query="My payment failed")
    result = graph.invoke(state)

    assert result["decision"] == "RESPOND"
    assert result["response"] == "Payment response"
    assert result["query"] == "My payment failed"
    assert result["retrieved_docs"] == []


def test_workflow_refund_issue_with_rag(monkeypatch):
    def intent_run(state):
        state.intent = ["REFUND_ISSUE"]
        state.intent_confidence = 0.9
        return state

    def emotion_run(state):
        state.emotion = "NEUTRAL"
        state.emotion_confidence = 0.8
        return state

    def decision_run(state):
        state.decision = "RESPOND"
        state.requires_rag = True
        return state

    def rag_run(state):
        state.retrieved_docs = [
            {"content": "Refund policy", "source": "ShopSphere_Refund_Policy.md"}
        ]
        return state

    def response_run(state):
        state.response = "Refund response"
        return state

    monkeypatch.setattr(graph_module, "intent_agent", StubAgent(intent_run))
    monkeypatch.setattr(graph_module, "emotion_agent", StubAgent(emotion_run))
    monkeypatch.setattr(graph_module, "decision_agent", StubAgent(decision_run))
    monkeypatch.setattr(graph_module, "rag_agent", StubAgent(rag_run))
    monkeypatch.setattr(graph_module, "response_agent", StubAgent(response_run))

    state = CustomerState(query="What is your refund policy?")
    result = graph.invoke(state)

    assert result["decision"] == "RESPOND"
    assert result["retrieved_docs"]
    assert result["response"] == "Refund response"


def test_workflow_unknown_intent_clarify(monkeypatch):
    def intent_run(state):
        state.intent = ["UNKNOWN_INTENT"]
        state.intent_confidence = 0.2
        return state

    def emotion_run(state):
        state.emotion = "NEUTRAL"
        state.emotion_confidence = 0.8
        return state

    def decision_run(state):
        state.decision = "CLARIFY"
        state.clarification_needed = True
        state.clarification_question = "Can you clarify?"
        return state

    def response_run(state):
        state.response = "Clarification response"
        return state

    monkeypatch.setattr(graph_module, "intent_agent", StubAgent(intent_run))
    monkeypatch.setattr(graph_module, "emotion_agent", StubAgent(emotion_run))
    monkeypatch.setattr(graph_module, "decision_agent", StubAgent(decision_run))
    monkeypatch.setattr(graph_module, "response_agent", StubAgent(response_run))

    state = CustomerState(query="Help")
    result = graph.invoke(state)

    assert result["decision"] == "CLARIFY"
    assert result["follow_up_required"] is True
    assert result["response"] == "Clarification response"


def test_workflow_repeated_complaint_escalation(monkeypatch):
    def intent_run(state):
        state.intent = ["DELIVERY_ISSUE"]
        state.intent_confidence = 0.9
        return state

    def emotion_run(state):
        state.emotion = "ANGRY"
        state.emotion_confidence = 0.9
        return state

    def decision_run(state):
        state.decision = "ESCALATE"
        state.priority = "HIGH"
        return state

    def escalation_run(state):
        state.escalated = True
        state.escalation_details = {
            "ticket_id": "TKT-999",
            "assigned_team": "Priority Support Team",
            "priority": "HIGH",
        }
        return state

    def response_run(state):
        state.response = "Escalation response"
        return state

    monkeypatch.setattr(graph_module, "intent_agent", StubAgent(intent_run))
    monkeypatch.setattr(graph_module, "emotion_agent", StubAgent(emotion_run))
    monkeypatch.setattr(graph_module, "decision_agent", StubAgent(decision_run))
    monkeypatch.setattr(graph_module, "escalation_agent", StubAgent(escalation_run))
    monkeypatch.setattr(graph_module, "response_agent", StubAgent(response_run))

    state = CustomerState(query="My delivery failed twice")
    result = graph.invoke(state)

    assert result["decision"] == "ESCALATE"
    assert result["escalated"] is True
    assert result["escalation_details"]["ticket_id"] == "TKT-999"
    assert result["response"] == "Escalation response"
