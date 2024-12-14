import sqlite3
from datetime import datetime
import json
from typing import List, Dict, Optional
from contextlib import contextmanager


class ConversationStore:
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the database and create necessary tables"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context TEXT,
                    metadata TEXT,
                    timestamp DATETIME NOT NULL,
                    CONSTRAINT idx_user_timestamp UNIQUE (user_id, timestamp)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id 
                ON conversations(user_id)
            """)

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()

    def add_message(self, user_id: str, message: str, response: str,
                    context: Optional[Dict] = None) -> None:
        """Add a new message to the conversation history"""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO conversations 
                (user_id, message, response, context, timestamp) 
                VALUES (?, ?, ?, ?, ?)""",
                (
                    user_id,
                    message,
                    response,
                    json.dumps(context) if context else None,
                    datetime.now().isoformat()
                )
            )

    def get_recent_messages(self, user_id: str, limit: int = 1) -> List[Dict]:
        """Get the most recent messages for a user"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT message, response, timestamp 
                FROM conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?""",
                (user_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_historical_messages(self, user_id: str, skip: int = 0,
                                limit: int = 10) -> List[Dict]:
        """Get historical messages for summarization"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT message, response, timestamp 
                FROM conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?""",
                (user_id, limit, skip)
            )
            return [dict(row) for row in cursor.fetchall()]

    def clean_old_messages(self, days_old: int = 30) -> None:
        """Clean up old messages to manage database size"""
        with self._get_connection() as conn:
            threshold = (datetime.now() - datetime.timedelta(days=days_old)).isoformat()
            conn.execute(
                "DELETE FROM conversations WHERE timestamp < ?",
                (threshold,)
            )