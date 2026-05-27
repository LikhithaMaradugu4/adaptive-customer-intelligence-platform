from typing import Any, Dict, List, Union

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


def _format_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Format messages into a role-prefixed string for prompts.
    """

    if not messages:

        return "No recent conversation."

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


def get_agent_context(
    state: StateLike,
    recent_messages_count: int = 6
) -> str:
    """
    Build lightweight agent context using summary + recent messages.
    """

    summary = _extract_summary(
        state
    )

    summary_text = (
        summary
        if summary.strip()
        else "No summary available."
    )

    messages = _extract_messages(
        state
    )

    recent_messages = (
        messages[-recent_messages_count:]
        if messages
        else []
    )

    recent_text = _format_messages(
        recent_messages
    )

    return (
        "Summary:\n"
        f"{summary_text}\n\n"
        "Recent Conversation:\n"
        f"{recent_text}"
    )
