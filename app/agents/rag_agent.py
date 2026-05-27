from app.state import CustomerState

from app.rag.retriever import (
    retrieve_documents
)

from app.utils.query_rewriter import (
    rewrite_query
)


from app.utils.reranker import (
    rerank_documents
)

from app.utils.helpers import (
    get_rag_reason,
    should_use_rag
)


class RAGAgent:
    """
    Enterprise-grade RAG Agent.

    Pipeline:
    1. Conversational query rewriting
    2. Semantic vector retrieval
    3. Intelligent reranking
    4. Retrieval diagnostics

    Goals:
    - maximize retrieval relevance
    - reduce hallucinations
    - improve conversational continuity
    - improve grounded responses
    """

    def __init__(self):

        # ---------------------------------
        # Retrieval config
        # ---------------------------------

        self.initial_k = 10

        self.final_k = 3

    # ---------------------------------
    # Build retrieval query
    # ---------------------------------

    def _build_query(
        self,
        state: CustomerState,
        conversation_context: str
    ):

        rewritten_query = (
            rewrite_query(

                query=state.query,

                conversation_context=(
                    conversation_context
                )
            )
        )

        return rewritten_query

    # ---------------------------------
    # Retrieve candidate docs
    # ---------------------------------

    def _retrieve_candidates(
        self,
        query: str,
        state: CustomerState
    ):

        retrieved_docs = (
            retrieve_documents(

                query=query,

                intents=state.intent,

                k=self.initial_k
            )
        )

        return retrieved_docs

    # ---------------------------------
    # Rerank retrieved docs
    # ---------------------------------

    def _rerank_documents(
        self,
        query: str,
        retrieved_docs
    ):

        reranked_docs = (
            rerank_documents(

                query=query,

                documents=retrieved_docs
            )
        )

        return reranked_docs[:self.final_k]

    # ---------------------------------
    # Build retrieval diagnostics
    # ---------------------------------

    def _build_retrieval_metadata(
        self,
        state: CustomerState,
        rewritten_query: str,
        retrieved_docs
    ):

        state.metadata[
            "rewritten_query"
        ] = rewritten_query

        state.metadata[
            "retrieved_documents"
        ] = len(
            retrieved_docs
        )

        state.metadata[
            "retrieval_source"
        ] = (
            "query_rewrite"
            "_semantic_search"
            "_reranking"
        )

        # ---------------------------------
        # Sources tracking
        # ---------------------------------

        sources = []

        for doc in retrieved_docs:

            source = doc.get(
                "source",
                "unknown"
            )

            if source not in sources:

                sources.append(source)

        state.metadata[
            "retrieved_sources"
        ] = sources

    # ---------------------------------
    # Main execution
    # ---------------------------------

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            if not should_use_rag(
                state.query,
                state.intent,
                state.decision,
                state.intent_confidence
            ):

                state.retrieved_docs = []

                state.metadata[
                    "retrieval_source"
                ] = "skipped_non_domain"

                state.metadata[
                    "rag_reason"
                ] = get_rag_reason(
                    state.query,
                    state.intent,
                    state.decision,
                    state.intent_confidence,
                    False
                )

                return state

            # ---------------------------------
            # Build lightweight retrieval context
            # ---------------------------------

            summary_only_context = (
                f"Summary:\n{state.summary}"
                if state.summary
                else "No summary available."
            )

            # ---------------------------------
            # Query rewriting
            # ---------------------------------

            rewritten_query = (
                self._build_query(

                    state,

                    summary_only_context
                )
            )

            print("\n" + "=" * 60)

            print("RAG PIPELINE")

            print("=" * 60)

            print("\nOriginal Query:")
            print(state.query)

            print("\nRewritten Query:")
            print(rewritten_query)

            # ---------------------------------
            # Candidate retrieval
            # ---------------------------------

            candidate_docs = (
                self._retrieve_candidates(

                    rewritten_query,

                    state
                )
            )

            print(
                f"\nRetrieved "
                f"{len(candidate_docs)} "
                f"candidate documents."
            )

            # ---------------------------------
            # Reranking
            # ---------------------------------

            reranked_docs = (
                self._rerank_documents(

                    rewritten_query,

                    candidate_docs
                )
            )

            print(
                f"\nReranked to "
                f"{len(reranked_docs)} "
                f"final documents."
            )

            # ---------------------------------
            # Debug retrieval preview
            # ---------------------------------

            print("\nFinal Retrieved Sources:")

            for idx, doc in enumerate(
                reranked_docs,
                start=1
            ):

                print(
                    f"\n[{idx}] "
                    f"{doc.get('source')}"
                )

                preview = (
                    doc.get(
                        "content",
                        ""
                    )[:250]
                )

                print(preview)

                print("-" * 40)

            # ---------------------------------
            # Update state
            # ---------------------------------

            state.retrieved_docs = (
                reranked_docs
            )

            # ---------------------------------
            # Metadata tracking
            # ---------------------------------

            self._build_retrieval_metadata(

                state,

                rewritten_query,

                reranked_docs
            )

            return state

        except Exception as e:

            print("\nRAGAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"RAGAgent Error: {str(e)}"
            )

            state.retry_count += 1

            # ---------------------------------
            # Safe fallback
            # ---------------------------------

            state.retrieved_docs = []

            state.metadata[
                "retrieval_source"
            ] = "failed"

            return state