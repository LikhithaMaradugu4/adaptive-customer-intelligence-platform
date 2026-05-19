import sqlite3


class DatabaseService:
    """
    SQLite database service.
    """

    def __init__(
        self,
        db_path: str = "customer_support.db"
    ):

        self.db_path = db_path

        self._initialize_database()

    def _initialize_database(self):
        """
        Create required tables.
        """

        connection = sqlite3.connect(
            self.db_path
        )

        cursor = connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            customer_id TEXT,

            query TEXT,

            intent TEXT,

            emotion TEXT,

            escalated INTEGER,

            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        connection.commit()
        connection.close()

    def save_conversation(
        self,
        customer_id: str,
        query: str,
        intent: str,
        emotion: str,
        escalated: bool
    ):
        """
        Store conversation record.
        """

        connection = sqlite3.connect(
            self.db_path
        )

        cursor = connection.cursor()

        cursor.execute("""
        INSERT INTO conversations (
            customer_id,
            query,
            intent,
            emotion,
            escalated
        )
        VALUES (?, ?, ?, ?, ?)
        """, (
            customer_id,
            query,
            intent,
            emotion,
            int(escalated)
        ))

        connection.commit()
        connection.close()

    def get_customer_history(
        self,
        customer_id: str,
        limit: int = 5
    ):
        """
        Retrieve recent customer history.
        """

        connection = sqlite3.connect(
            self.db_path
        )

        cursor = connection.cursor()

        cursor.execute("""
        SELECT
            query,
            intent,
            emotion,
            escalated,
            timestamp
        FROM conversations
        WHERE customer_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """, (
            customer_id,
            limit
        ))

        rows = cursor.fetchall()

        connection.close()

        history = []

        for row in rows:

            history.append({
                "query": row[0],
                "intent": row[1],
                "emotion": row[2],
                "escalated": bool(row[3]),
                "timestamp": row[4]
            })

        return history


# Singleton instance
database_service = DatabaseService()