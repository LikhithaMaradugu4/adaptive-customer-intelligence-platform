from langchain_community.embeddings import (
    FastEmbedEmbeddings
)

from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"

embedding_model = FastEmbedEmbeddings()

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)


def retrieve_documents(
    query: str,
    k: int = 3
):
    """
    Retrieve top-k relevant documents.
    """

    results = vector_store.similarity_search(
        query,
        k=k
    )

    retrieved_docs = []

    for doc in results:

        retrieved_docs.append({
            "content": doc.page_content,
            "source": doc.metadata.get(
                "source",
                "unknown"
            )
        })

    return retrieved_docs