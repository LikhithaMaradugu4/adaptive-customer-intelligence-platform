from app.services.llm_service import (
    llm_service
)


def rerank_documents(
    query: str,
    documents: list
):

    try:

        llm = llm_service._create_llm(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0
        )

        scored_docs = []

        for doc in documents:

            prompt = f"""
Rate how relevant this document is
to the customer query.

Score between 1 and 10.

Customer Query:
{query}

Document:
{doc.page_content}

Return only numeric score.
"""

            response = llm.invoke(prompt)

            try:

                score = float(
                    response.content.strip()
                )

            except:

                score = 0

            scored_docs.append(
                (score, doc)
            )

        scored_docs.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        return [
            doc
            for score, doc
            in scored_docs[:3]
        ]

    except Exception as e:

        print("\nReranker ERROR:")
        print(str(e))

        return documents[:3]