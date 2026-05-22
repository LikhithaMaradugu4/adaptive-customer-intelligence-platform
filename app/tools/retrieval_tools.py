from langchain.tools import tool

from app.rag.retriever import (
    retrieve_documents
)


@tool
def retrieve_policy_documents(
    query: str
):
    """
    Retrieve relevant company policy
    and support documents.
    """

    documents = retrieve_documents(
        query=query,
        k=3
    )

    return documents