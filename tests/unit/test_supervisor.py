from app.state import CustomerState
from app.supervisor import Supervisor


class DummyAgent:
    def run(self, state):
        state.intent = ["PAYMENT_ISSUE"]
        state.intent_confidence = 0.9
        return state


class InvalidAgent:
    def run(self, state):
        state.intent = None
        state.intent_confidence = None
        return state


class TimeoutAgent:
    def run(self, _state):
        raise TimeoutError("timeout")


def test_supervisor_runs_agent_successfully():
    supervisor = Supervisor()
    state = CustomerState(query="payment")

    result = supervisor.run_agent(
        agent_callable=DummyAgent().run,
        state=state,
        validator=supervisor.validate_intent,
    )

    assert result.metadata["last_successful_agent"] == "DummyAgent"
    assert result.intent == ["PAYMENT_ISSUE"]


def test_supervisor_validation_failure_escalates():
    supervisor = Supervisor()
    state = CustomerState(query="payment")

    result = supervisor.run_agent(
        agent_callable=InvalidAgent().run,
        state=state,
        validator=supervisor.validate_intent,
    )

    assert result.escalated is True
    assert result.escalation_reason == "agent_failure"
    assert any("Validation failed" in err for err in result.errors)


def test_supervisor_timeout_handling_escalates():
    supervisor = Supervisor()
    state = CustomerState(query="timeout")

    result = supervisor.run_agent(
        agent_callable=TimeoutAgent().run,
        state=state,
        validator=None,
    )

    assert result.escalated is True
    assert result.escalation_reason == "agent_failure"
    assert result.retry_count == 2
    assert any("Supervisor Error" in err for err in result.errors)
