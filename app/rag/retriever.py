from langchain_community.embeddings import (
    FastEmbedEmbeddings
)

from langchain_chroma import Chroma


# ---------------------------------
# Vector DB path
# ---------------------------------

CHROMA_PATH = "chroma_db"

# ---------------------------------
# Embedding model
# ---------------------------------

embedding_model = FastEmbedEmbeddings()

# ---------------------------------
# Load existing vector DB
# ---------------------------------

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

# ---------------------------------
# Intent-aware retrieval filters
# ---------------------------------

INTENT_FILTERS = {

    "REFUND_ISSUE": [
        "refund"
    ],

    "RETURN_ISSUE": [
        "return"
    ],

    "DELIVERY_ISSUE": [
        "delivery",
        "shipping"
    ],

    "PRODUCT_ISSUE": [
        "product",
        "warranty"
    ],

    "PAYMENT_ISSUE": [
        "payment"
    ],

    "ACCOUNT_ISSUE": [
        "account"
    ]
}


def retrieve_documents(
    query: str,
    intents=None,
    k: int = 3
):
    """
    Intent-aware semantic retrieval.
    """

    results = vector_store.similarity_search(
        query,
        k=10
    )

    filtered_results = []

    # ---------------------------------
    # Intent-aware filtering
    # ---------------------------------

    if intents:

        allowed_keywords = []

        for intent in intents:

            allowed_keywords.extend(
                INTENT_FILTERS.get(
                    intent,
                    []
                )
            )

        for doc in results:

            content = (
                doc.page_content.lower()
            )

            source = (
                doc.metadata.get(
                    "source",
                    ""
                ).lower()
            )

            if any(
                keyword in content
                or keyword in source
                for keyword in allowed_keywords
            ):

                filtered_results.append(
                    doc
                )

    # ---------------------------------
    # Fallback
    # ---------------------------------

    if not filtered_results:

        filtered_results = results[:k]

    retrieved_docs = []

    for doc in filtered_results[:k]:

        retrieved_docs.append({

            "content": doc.page_content,

            "source": doc.metadata.get(
                "source",
                "unknown"
            )
        })

    return retrieved_docs