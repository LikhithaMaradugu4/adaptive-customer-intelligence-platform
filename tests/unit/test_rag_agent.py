from app.agents.rag_agent import RAGAgent
from app.state import CustomerState


def test_rag_agent_retrieves_documents(monkeypatch):
    agent = RAGAgent()
    monkeypatch.setattr(
        "app.agents.rag_agent.retrieve_documents",
        lambda query, intents, k: [
            {"content": "Refund policy", "source": "ShopSphere_Refund_Policy.md"}
        ],
    )

    state = CustomerState(query="What is your refund policy?")
    state.intent = ["REFUND_ISSUE"]

    result = agent.run(state)

    assert result.retrieved_docs
    assert result.metadata["retrieved_documents"] == 1


def test_rag_agent_handles_no_documents(monkeypatch):
    agent = RAGAgent()
    monkeypatch.setattr(
        "app.agents.rag_agent.retrieve_documents",
        lambda query, intents, k: [],
    )

    state = CustomerState(query="unknown")
    result = agent.run(state)

    assert result.retrieved_docs == []
    assert result.metadata["retrieved_documents"] == 0


def test_rag_agent_handles_errors(monkeypatch):
    agent = RAGAgent()
    monkeypatch.setattr(
        "app.agents.rag_agent.retrieve_documents",
        lambda query, intents, k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    state = CustomerState(query="broken")
    result = agent.run(state)

    assert result.retrieved_docs == []
    assert result.retry_count == 1
    assert any("RAGAgent Error" in err for err in result.errors)
