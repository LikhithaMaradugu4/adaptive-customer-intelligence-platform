def build_conversation_context(
    conversation_history,
    max_messages: int = 6
):
    """
    Build conversational context window.
    """

    if not conversation_history:

        return "No previous conversation."

    formatted_history = []

    for item in conversation_history[-max_messages:]:

        role = item.get(
            "role",
            "unknown"
        )

        message = item.get(
            "message",
            ""
        )

        formatted_history.append(
            f"{role}: {message}"
        )

    return "\n".join(
        formatted_history
    )