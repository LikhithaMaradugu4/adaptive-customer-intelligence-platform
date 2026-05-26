from app.state import CustomerState

from app.rag.retriever import (
    retrieve_documents
)

from app.utils.query_rewriter import (
    rewrite_query
)

from app.utils.conversation import (
    build_conversation_context
)


class RAGAgent:
    """
    Enterprise RAG Agent

    Pipeline:
    - Query rewriting
    - Semantic retrieval
    - Reranking
    """

    def __init__(self):

        self.top_k = 3

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Build conversation context
            # ---------------------------------

            conversation_context = (
                build_conversation_context(
                    state.conversation_history
                )
            )

            # ---------------------------------
            # Query rewriting
            # ---------------------------------

            rewritten_query = (
                rewrite_query(
                    query=state.query,
                    conversation_context=(
                        conversation_context
                    )
                )
            )

            print("\nOriginal Query:")
            print(state.query)

            print("\nRewritten Query:")
            print(rewritten_query)

            # ---------------------------------
            # Retrieve documents
            # ---------------------------------

            retrieved_docs = (
                retrieve_documents(
                    query=rewritten_query,
                    k=self.top_k
                )
            )

            print(
                f"\nRetrieved "
                f"{len(retrieved_docs)} "
                f"documents."
            )

            # ---------------------------------
            # Update state
            # ---------------------------------

            state.retrieved_docs = (
                retrieved_docs
            )

            state.metadata[
                "retrieved_documents"
            ] = len(
                retrieved_docs
            )

            state.metadata[
                "rewritten_query"
            ] = rewritten_query

            state.metadata[
                "retrieval_source"
            ] = "reranked_vector_search"

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