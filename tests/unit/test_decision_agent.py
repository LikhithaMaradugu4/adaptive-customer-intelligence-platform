from app.agents.decision_agent import DecisionAgent
from app.schemas import DecisionOutput
from app.state import CustomerState


def test_decision_agent_premium_angry_escalates():
    agent = DecisionAgent()
    state = CustomerState(query="bad experience")
    state.intent = ["DELIVERY_ISSUE"]
    state.emotion = "ANGRY"
    state.customer_profile = {"profile_type": "PREMIUM"}

    result = agent.run(state)

    assert result.decision == "ESCALATE"
    assert result.priority == "HIGH"
    assert result.metadata["decision_source"] == "business_rules"


def test_decision_agent_high_risk_requires_approval():
    agent = DecisionAgent()
    state = CustomerState(query="refund issue")
    state.intent = ["REFUND_ISSUE"]
    state.emotion = "NEUTRAL"
    state.customer_profile = {"profile_type": "HIGH_RISK"}

    result = agent.run(state)

    assert result.decision == "FRAUD_REVIEW"
    assert result.human_approval_required is True
    assert result.approval_reason == "high_risk_customer"


def test_decision_agent_repeated_complaint_escalates():
    agent = DecisionAgent()
    state = CustomerState(query="delivery issue again")
    state.intent = ["DELIVERY_ISSUE"]
    state.emotion = "FRUSTRATED"
    state.customer_history = [
        {"intent": "DELIVERY_ISSUE"},
        {"intent": "DELIVERY_ISSUE"},
    ]

    result = agent.run(state)

    assert result.decision == "ESCALATE"
    assert result.priority == "HIGH"


def test_decision_agent_unknown_intent_clarifies():
    agent = DecisionAgent()
    state = CustomerState(query="??")
    state.intent = ["UNKNOWN_INTENT"]
    state.emotion = "NEUTRAL"

    result = agent.run(state)

    assert result.decision == "CLARIFY"
    assert result.clarification_needed is True
    assert "please explain" in result.clarification_question.lower()


def test_decision_agent_llm_response_path(monkeypatch):
    agent = DecisionAgent()
    state = CustomerState(query="What is the return policy?")
    state.intent = ["GENERAL_QUERY"]
    state.emotion = "NEUTRAL"

    monkeypatch.setattr(
        agent,
        "_llm_decision",
        lambda _state: DecisionOutput(
            decision="RESPOND",
            priority="NORMAL",
            clarification_needed=False,
            clarification_question=None,
            human_approval_required=False,
            approval_reason=None,
            requires_rag=False,
        ),
    )

    result = agent.run(state)

    assert result.decision == "RESPOND"
    assert result.metadata["decision_source"] == "tool_calling_llm"
