from langchain.tools import tool

from app.rag.retriever import (
    retrieve_documents
)


@tool
def retrieve_policy_documents(
    query: str
) -> str:
    """
    Retrieve relevant support
    policy documents.
    """

    docs = retrieve_documents(
        query=query
    )

    formatted_docs = []

    for idx, doc in enumerate(
        docs,
        start=1
    ):

        formatted_docs.append(

            f"""
Document {idx}

Source:
{doc['source']}

Content:
{doc['content']}
"""
        )

    return "\n".join(
        formatted_docs
    )