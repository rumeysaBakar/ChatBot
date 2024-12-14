import logging
from typing import Dict, List, Any
import asyncio

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, settings):
        self.settings = settings
        self._history = {}  # Simple in-memory storage for demo

    async def get_context(self, user_id: str) -> Dict[str, Any]:
        """Get conversation context including summary and recent messages"""
        try:
            history = self._history.get(user_id, [])
            if not history:
                return {
                    'summary': '',
                    'recent_messages': []
                }

            # For demo, return last few messages
            recent_messages = history[-3:]

            # Simple summary (in production, you'd want more sophisticated summarization)
            summary = f"Previous conversation included {len(history)} messages about: " + \
                      ", ".join([msg["message"][:30] + "..." for msg in history[-2:]])

            return {
                'summary': summary,
                'recent_messages': recent_messages
            }
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return {'summary': '', 'recent_messages': []}

    async def add_message(self, user_id: str, message: str, response: str, context: Dict):
        """Add a new message to the conversation history"""
        try:
            if user_id not in self._history:
                self._history[user_id] = []

            self._history[user_id].append({
                "message": message,
                "response": response,
                "context": context
            })

            # Limit history size
            if len(self._history[user_id]) > 100:
                self._history[user_id] = self._history[user_id][-100:]

        except Exception as e:
            logger.error(f"Error adding message to history: {str(e)}")

    async def close(self):
        """Cleanup resources if needed"""
        self._history.clear()