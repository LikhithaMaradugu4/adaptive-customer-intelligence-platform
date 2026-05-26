from langchain_community.embeddings import (
    FastEmbedEmbeddings
)

from langchain_chroma import Chroma

from app.utils.reranker import (
    rerank_documents
)

# ---------------------------------
# Vector DB path
# ---------------------------------

CHROMA_PATH = "chroma_db"

# ---------------------------------
# Embedding model
# ---------------------------------

embedding_model = (
    FastEmbedEmbeddings()
)

# ---------------------------------
# Load vector DB
# ---------------------------------

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)


def retrieve_documents(
    query: str,
    k: int = 3
):
    """
    Enterprise retrieval pipeline.

    - Vector retrieval
    - Reranking
    """

    # ---------------------------------
    # Candidate retrieval
    # ---------------------------------

    candidate_docs = (
        vector_store.similarity_search(
            query,
            k=8
        )
    )

    # ---------------------------------
    # Rerank
    # ---------------------------------

    reranked_docs = (
        rerank_documents(
            query=query,
            documents=candidate_docs
        )
    )

    retrieved_docs = []

    # ---------------------------------
    # Final formatting
    # ---------------------------------

    for doc in reranked_docs[:k]:

        retrieved_docs.append({

            "content": (
                doc.page_content
            ),

            "source": (
                doc.metadata.get(
                    "source_file",
                    "unknown"
                )
            ),

            "page": (
                doc.metadata.get(
                    "page",
                    "unknown"
                )
            ),

            "category": (
                doc.metadata.get(
                    "category",
                    "unknown"
                )
            )
        })

    return retrieved_docs