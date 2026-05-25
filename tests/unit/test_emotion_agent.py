from app.agents.emotion_agent import EmotionAgent
from app.state import CustomerState


def test_emotion_agent_high_confidence_uses_vader(monkeypatch):
    agent = EmotionAgent()
    monkeypatch.setattr(agent, "_predict_with_vader", lambda _q: ("SATISFIED", 0.9))

    state = CustomerState(query="Thanks, great service!")
    result = agent.run(state)

    assert result.emotion == "SATISFIED"
    assert result.emotion_confidence == 0.9
    assert result.metadata["emotion_source"] == "vader"


def test_emotion_agent_low_confidence_uses_llm_fallback(monkeypatch):
    agent = EmotionAgent()
    monkeypatch.setattr(agent, "_predict_with_vader", lambda _q: ("NEUTRAL", 0.2))
    monkeypatch.setattr(agent, "_predict_with_llm", lambda _q, _h: ("FRUSTRATED", 0.7))

    state = CustomerState(query="This is getting annoying")
    result = agent.run(state)

    assert result.emotion == "FRUSTRATED"
    assert result.emotion_confidence == 0.7
    assert result.metadata["emotion_source"] == "llm_fallback"


def test_emotion_agent_handles_errors(monkeypatch):
    agent = EmotionAgent()
    monkeypatch.setattr(agent, "_predict_with_vader", lambda _q: (_ for _ in ()).throw(RuntimeError("boom")))

    state = CustomerState(query="error case")
    result = agent.run(state)

    assert result.emotion == "NEUTRAL"
    assert result.emotion_confidence == 0.0
    assert result.retry_count == 1
    assert any("EmotionAgent Error" in err for err in result.errors)
