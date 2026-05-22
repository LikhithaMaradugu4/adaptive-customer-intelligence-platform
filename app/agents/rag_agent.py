from app.state import CustomerState

from app.rag.retriever import (
    retrieve_documents
)


class RAGAgent:
    """
    Retrieval-Augmented Generation Agent.

    Retrieves relevant company knowledge
    using semantic search.
    """

    def __init__(self):

        self.top_k = 3

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            query = state.query

            retrieved_docs = (
                retrieve_documents(
                    query=query,
                    intents=state.intent,
                    k=self.top_k
                )
            )

            state.retrieved_docs = (
                retrieved_docs
            )

            state.metadata[
                "retrieved_documents"
            ] = len(retrieved_docs)

            return state

        except Exception as e:

            print("\nRAGAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"RAGAgent Error: {str(e)}"
            )

            state.retry_count += 1

            state.retrieved_docs = []

            return state