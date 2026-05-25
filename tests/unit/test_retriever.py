from app.rag import retriever


class FakeDoc:
    def __init__(self, content, source):
        self.page_content = content
        self.metadata = {"source": source}


def test_retriever_fallback_when_no_intent_match(monkeypatch):
    docs = [
        FakeDoc("general policy", "general.md"),
        FakeDoc("shipping details", "shipping.md"),
    ]

    monkeypatch.setattr(
        retriever.vector_store,
        "similarity_search",
        lambda _query, k=10: docs,
    )

    results = retriever.retrieve_documents(
        query="refund question",
        intents=["REFUND_ISSUE"],
        k=2,
    )

    assert len(results) == 2
    assert results[0]["content"] == "general policy"


def test_retriever_no_documents(monkeypatch):
    monkeypatch.setattr(
        retriever.vector_store,
        "similarity_search",
        lambda _query, k=10: [],
    )

    results = retriever.retrieve_documents(
        query="anything",
        intents=None,
        k=3,
    )

    assert results == []
