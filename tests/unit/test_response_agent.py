from app.agents.response_agent import ResponseAgent
from app.services.llm_service import llm_service
from app.state import CustomerState
from app.schemas import ResponseOutput
from tests.fixtures.sample_data import SAMPLE_DOCS, SAMPLE_TOOL_OUTPUTS


def test_response_agent_clarification_response(monkeypatch, prompt_capture):
    agent = ResponseAgent()

    def fake_llm(agent_name=None, temperature=0.0):
        return type(
        "LLM",
        (),
        {
            "with_structured_output": lambda self, _schema: type(
                "Structured",
                (),
                {"invoke": lambda _self, prompt: ResponseOutput(response="Need more details")},
            )(),
        },
        )()

    monkeypatch.setattr(llm_service, "get_llm_for_agent", fake_llm)

    state = CustomerState(query="?")
    state.clarification_needed = True
    state.clarification_question = "Can you уточнить?"

    result = agent.run(state)

    assert result.response == "Need more details"
    assert result.metadata["response_source"] == "clarification"


def test_response_agent_escalation_response(monkeypatch, fake_llm, prompt_capture):
    agent = ResponseAgent()
    monkeypatch.setattr(llm_service, "_create_llm", lambda **_kwargs: fake_llm)

    state = CustomerState(query="Angry about refund")
    state.decision = "ESCALATE"
    state.emotion = "ANGRY"
    state.escalation_details = {
        "ticket_id": "TKT-123",
        "assigned_team": "Priority Support Team",
    }

    result = agent.run(state)

    assert result.response == "Test response"
    assert result.metadata["response_source"] == "llm"
    assert "TKT-123" in prompt_capture["prompt"]
    assert "Priority Support Team" in prompt_capture["prompt"]


def test_response_agent_tone_adaptation_prompt_includes_emotion(monkeypatch, fake_llm, prompt_capture):
    agent = ResponseAgent()
    monkeypatch.setattr(llm_service, "_create_llm", lambda **_kwargs: fake_llm)

    state = CustomerState(query="Where is my order?")
    state.decision = "RESPOND"
    state.emotion = "ANGRY"
    state.intent = ["DELIVERY_ISSUE"]
    state.retrieved_docs = SAMPLE_DOCS
    state.tool_outputs = SAMPLE_TOOL_OUTPUTS

    result = agent.run(state)

    assert result.response == "Test response"
    assert "Customer Emotion:" in prompt_capture["prompt"]
    assert "ANGRY" in prompt_capture["prompt"]
