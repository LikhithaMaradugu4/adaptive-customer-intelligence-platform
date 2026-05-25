import joblib

from app.agents.intent_agent import IntentAgent
from app.state import CustomerState


class FakePipeline:
    classes_ = ["PAYMENT_ISSUE", "REFUND_ISSUE"]

    def predict_proba(self, _inputs):
        return [[0.1, 0.9]]


def test_intent_agent_high_confidence_uses_ml(monkeypatch):
    monkeypatch.setattr(joblib, "load", lambda _path: FakePipeline())
    agent = IntentAgent()
    monkeypatch.setattr(agent, "_predict_with_ml", lambda _q: ("PAYMENT_ISSUE", 0.9))

    state = CustomerState(query="Payment failed")
    result = agent.run(state)

    assert result.intent == ["PAYMENT_ISSUE"]
    assert result.intent_confidence == 0.9
    assert result.metadata["intent_source"] == "ml"


def test_intent_agent_low_confidence_uses_llm_fallback(monkeypatch):
    monkeypatch.setattr(joblib, "load", lambda _path: FakePipeline())
    agent = IntentAgent()
    monkeypatch.setattr(agent, "_predict_with_ml", lambda _q: ("PAYMENT_ISSUE", 0.2))
    monkeypatch.setattr(agent, "_predict_with_llm", lambda _q, _h: (["REFUND_ISSUE"], 0.8))

    state = CustomerState(query="I want a refund")
    result = agent.run(state)

    assert result.intent == ["REFUND_ISSUE"]
    assert result.intent_confidence == 0.8
    assert result.metadata["intent_source"] == "llm_fallback"


def test_intent_agent_multi_intent_output(monkeypatch):
    monkeypatch.setattr(joblib, "load", lambda _path: FakePipeline())
    agent = IntentAgent()
    monkeypatch.setattr(agent, "_predict_with_ml", lambda _q: ("PAYMENT_ISSUE", 0.1))
    monkeypatch.setattr(
        agent,
        "_predict_with_llm",
        lambda _q, _h: (["PAYMENT_ISSUE", "DELIVERY_ISSUE"], 0.7),
    )

    state = CustomerState(query="Payment failed and delivery is late")
    result = agent.run(state)

    assert result.intent == ["PAYMENT_ISSUE", "DELIVERY_ISSUE"]
    assert result.intent_confidence == 0.7


def test_intent_agent_handles_errors(monkeypatch):
    monkeypatch.setattr(joblib, "load", lambda _path: FakePipeline())
    agent = IntentAgent()
    monkeypatch.setattr(agent, "_predict_with_ml", lambda _q: (_ for _ in ()).throw(RuntimeError("boom")))

    state = CustomerState(query="something")
    result = agent.run(state)

    assert result.intent == ["UNKNOWN_INTENT"]
    assert result.intent_confidence == 0.0
    assert result.retry_count == 1
    assert any("IntentAgent Error" in err for err in result.errors)
