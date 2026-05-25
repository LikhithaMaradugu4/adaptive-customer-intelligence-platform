import joblib
import pytest

from app.agents.intent_agent import IntentAgent
from app.state import CustomerState


class FakePipeline:
    classes_ = ["GENERAL_QUERY"]

    def predict_proba(self, _inputs):
        return [[1.0]]


@pytest.mark.parametrize(
    "query",
    [
        "",
        "A" * 5000,
        "{malformed-json: true",
        "こんにちは、返金できますか？",
        "'; DROP TABLE users; --",
        "!!!@@@###$$$",
    ],
)
def test_intent_agent_handles_edge_queries(monkeypatch, query):
    monkeypatch.setattr(joblib, "load", lambda _path: FakePipeline())
    agent = IntentAgent()
    monkeypatch.setattr(agent, "_predict_with_ml", lambda _q: ("GENERAL_QUERY", 0.95))

    state = CustomerState(query=query)
    result = agent.run(state)

    assert result.intent == ["GENERAL_QUERY"]
    assert result.intent_confidence == 0.95
    assert result.errors == []
