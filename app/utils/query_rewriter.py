from app.services.llm_service import (
    llm_service
)


def rewrite_query(
    query: str,
    conversation_context: str
) -> str:

    try:

        prompt = f"""
You are a search query rewriting system.

Rewrite the customer query into
a standalone retrieval-friendly query.

IMPORTANT:
- Resolve references like:
  "it", "that", "earlier"
- Preserve original meaning
- Make query explicit
- Keep concise

Conversation Context:
{conversation_context}

Customer Query:
{query}
"""

        llm = llm_service._create_llm(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0
        )

        response = llm.invoke(
            prompt
        )

        return response.content.strip()

    except Exception as e:

        print("\nQuery Rewrite ERROR:")
        print(str(e))

        return query