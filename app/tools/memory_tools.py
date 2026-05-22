from langchain.tools import tool

from app.services.session_service import (
    session_service
)


@tool
def get_recent_messages(
    session_id: str
):
    """
    Fetch recent conversation messages
    from a session.
    """

    messages = (
        session_service
        .get_session_messages(
            session_id
        )
    )

    recent_messages = messages[-10:]

    return recent_messages


@tool
def get_customer_sessions(
    customer_id: str
):
    """
    Fetch all sessions
    for a customer.
    """

    sessions = (
        session_service
        .get_customer_sessions(
            customer_id
        )
    )

    return sessions