import asyncio
from typing import List, Dict
from config.settings import Settings
from utils.async_utils import AsyncSessionManager


class ConversationSummarizer(AsyncSessionManager):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._summary_cache = {}
        self._summary_lock = asyncio.Lock()

    async def get_summary(self, messages: List[Dict[str, str]]) -> str:
        """Generate a summary of the conversation history"""
        cache_key = self._get_cache_key(messages)

        async with self._summary_lock:
            # Check cache first
            if cache_key in self._summary_cache:
                return self._summary_cache[cache_key]

            # Generate new summary
            summary = await self._generate_summary(messages)
            self._update_cache(cache_key, summary)
            return summary

    def _get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate a cache key for the messages"""
        return hash(str(messages))

    def _update_cache(self, key: str, summary: str) -> None:
        """Update the summary cache and manage its size"""
        self._summary_cache[key] = summary

        # Clean cache if it gets too large
        if len(self._summary_cache) > self.settings.MAX_CACHE_ITEMS:
            # Remove oldest entries
            oldest_keys = sorted(self._summary_cache.keys())[:-self.settings.CACHE_CLEANUP_THRESHOLD]
            for old_key in oldest_keys:
                del self._summary_cache[old_key]

    async def _generate_summary(self, messages: List[Dict[str, str]]) -> str:
        """Generate a summary using the LLM"""
        async with self._get_session() as session:
            conversation_text = self._format_conversation(messages)

            payload = {
                "model": self.settings.MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "Create a brief, focused summary of the key points in this conversation."
                    },
                    {
                        "role": "user",
                        "content": conversation_text
                    }
                ],
                "temperature": self.settings.SUMMARY_TEMPERATURE,
                "max_tokens": 200
            }

            try:
                async with session.post(
                        f"{self.settings.NVIDIA_BASE_URL}/chat/completions",
                        json=payload
                ) as response:
                    if not response.ok:
                        raise Exception(f"Summary generation error: {await response.text()}")

                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

            except Exception as e:
                print(f"Error generating summary: {str(e)}")
                return "Error generating conversation summary."

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for summarization"""
        formatted_messages = []
        for msg in messages:
            if 'message' in msg and 'response' in msg:
                formatted_messages.extend([
                    f"User: {msg['message']}",
                    f"Assistant: {msg['response']}"
                ])
        return "\n".join(formatted_messages)
