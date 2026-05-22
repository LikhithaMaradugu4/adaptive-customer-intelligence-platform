import uuid

from datetime import datetime

from app.database.mongo import (
    mongodb
)


class SessionService:
    """
    Session persistence layer.
    """

    def __init__(self):

        self.sessions_collection = (
            mongodb.get_collection(
                "sessions"
            )
        )

        self.messages_collection = (
            mongodb.get_collection(
                "messages"
            )
        )

    # ---------------------------------
    # Create new session
    # ---------------------------------

    def create_session(
        self,
        customer_id: str,
        title: str = "New Chat"
    ):

        session_id = str(
            uuid.uuid4()
        )

        session_data = {
            "session_id": session_id,
            "customer_id": customer_id,
            "title": title,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "status": "ACTIVE"
        }

        self.sessions_collection.insert_one(
            session_data
        )
        
        session_data.pop("_id", None)

        return session_data

    # ---------------------------------
    # Get session by ID
    # ---------------------------------

    def get_session(
        self,
        session_id: str
    ):

        session = (
            self.sessions_collection.find_one(
                {
                    "session_id": session_id
                },
                {
                    "_id": 0
                }
            )
        )

        return session

    # ---------------------------------
    # Get customer sessions
    # ---------------------------------

    def get_customer_sessions(
        self,
        customer_id: str
    ):

        sessions = list(
            self.sessions_collection.find(
                {
                    "customer_id": customer_id
                },
                {
                    "_id": 0
                }
            ).sort(
                "updated_at",
                -1
            )
        )

        return sessions

    # ---------------------------------
    # Save message
    # ---------------------------------

    def save_message(
        self,
        session_id: str,
        customer_id: str,
        role: str,
        message: str
    ):

        message_data = {
            "session_id": session_id,
            "customer_id": customer_id,
            "role": role,
            "message": message,
            "timestamp": datetime.utcnow()
        }

        self.messages_collection.insert_one(
            message_data
        )

        # ---------------------------------
        # Update session timestamp
        # ---------------------------------

        self.sessions_collection.update_one(
            {
                "session_id": session_id
            },
            {
                "$set": {
                    "updated_at": datetime.utcnow()
                }
            }
        )

    # ---------------------------------
    # Get session messages
    # ---------------------------------

    def get_session_messages(
        self,
        session_id: str
    ):

        messages = list(
            self.messages_collection.find(
                {
                    "session_id": session_id
                },
                {
                    "_id": 0
                }
            ).sort(
                "timestamp",
                1
            )
        )

        return messages


# ---------------------------------
# Singleton instance
# ---------------------------------

session_service = SessionService()