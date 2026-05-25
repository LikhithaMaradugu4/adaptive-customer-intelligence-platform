from pydantic import BaseModel

from app.services.llm_service import (
    llm_service
)


class TitleOutput(BaseModel):

    title: str


def generate_session_title(
    query: str
) -> str:
    """
    Generate concise session title
    using LLM.
    """

    try:

        prompt = f"""
Generate a concise customer support
chat title.

Rules:
- Maximum 5 words
- Professional
- No quotes
- No punctuation at end
- Summarize core issue/topic

Customer Query:
{query}
"""

        llm = llm_service._create_llm(
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )

        structured_llm = (
            llm.with_structured_output(
                TitleOutput
            )
        )

        response = structured_llm.invoke(
            prompt
        )

        title = response.title.strip()

        # ---------------------------------
        # Safety fallback
        # ---------------------------------

        if not title:

            return "Customer Support Chat"

        return title

    except Exception as e:

        print("\nTitle Generator ERROR:")
        print(str(e))

        return "Customer Support Chat"