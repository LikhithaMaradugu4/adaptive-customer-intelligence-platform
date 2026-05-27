from typing import Any, Dict, List, Union

from app.services.llm_service import llm_service
from app.state import CustomerState


StateLike = Union[CustomerState, Dict[str, Any]]


def _extract_messages(state: StateLike) -> List[Dict[str, Any]]:
    """
    Extract conversation messages from either a model or dict state.
    """

    if isinstance(state, dict):

        return state.get(
            "conversation_history",
            []
        )

    return state.conversation_history


def _extract_summary(state: StateLike) -> str:
    """
    Extract summary text from either a model or dict state.
    """

    if isinstance(state, dict):

        return state.get(
            "summary",
            ""
        ) or ""

    return state.summary or ""


def _assign_summary(state: StateLike, summary: str) -> None:
    """
    Store summary into either a model or dict state.
    """

    if isinstance(state, dict):

        state["summary"] = summary

    else:

        state.summary = summary


def _assign_messages(
    state: StateLike,
    messages: List[Dict[str, Any]]
) -> None:
    """
    Store messages into either a model or dict state.
    """

    if isinstance(state, dict):

        state["conversation_history"] = messages

    else:

        state.conversation_history = messages


def _format_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Format messages into a role-prefixed string for summarization.
    """

    if not messages:

        return "No previous conversation."

    formatted_lines = []

    for item in messages:

        role = item.get(
            "role",
            "unknown"
        )

        message = (
            item.get("message")
            or item.get("content")
            or ""
        )

        formatted_lines.append(
            f"{role}: {message}"
        )

    return "\n".join(
        formatted_lines
    )


def _normalize_llm_response(response: Any) -> str:
    """
    Normalize LLM responses into a summary string.
    """

    if hasattr(
        response,
        "content"
    ):

        return response.content.strip()

    return str(response).strip()


def update_conversation_summary(
    state: StateLike,
    threshold: int = 12,
    recent_messages_count: int = 6
) -> StateLike:
    """
    Summarize older messages and keep only recent ones in state.
    """

    messages = _extract_messages(
        state
    )

    if len(messages) <= threshold:

        return state

    # ---------------------------------
    # Split older and recent messages
    # ---------------------------------

    old_messages = messages[:-recent_messages_count]
    recent_messages = messages[-recent_messages_count:]

    if not old_messages:

        _assign_messages(
            state,
            recent_messages
        )

        return state

    previous_summary = _extract_summary(
        state
    )

    # ---------------------------------
    # Summarize older conversation
    # ---------------------------------

    prompt = f"""
You are a conversation summarization system.

Summarize key facts, unresolved issues, decisions, and identifiers.
Be concise and avoid speculation.

Previous Summary:
{previous_summary or "None"}

Conversation to Summarize:
{_format_messages(old_messages)}

Return a concise summary.
"""

    summary_response = (
        llm_service
        .invoke_with_fallback(

            agent_name="response",

            prompt=prompt,

            temperature=0.2
        )
    )

    new_summary = _normalize_llm_response(
        summary_response
    )

    if previous_summary:

        combined_summary = (
            f"{previous_summary.strip()}\n"
            f"{new_summary}"
        )

    else:

        combined_summary = new_summary

    _assign_summary(
        state,
        combined_summary
    )

    _assign_messages(
        state,
        recent_messages
    )

    return state
