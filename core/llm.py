import aiohttp
import json
import logging
import asyncio
from typing import List, Dict, Optional
from config.settings import Settings

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM interactions with proper session handling"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the LLM manager and create session"""
        if self._initialized:
            return

        try:
            logger.info("Initializing LLM Manager...")
            async with self._lock:
                if not self._initialized:
                    self.session = aiohttp.ClientSession(
                        headers=self.settings.headers,
                        timeout=aiohttp.ClientTimeout(total=self.settings.REQUEST_TIMEOUT)
                    )
                    self._initialized = True
            logger.info("LLM Manager initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing LLM Manager: {str(e)}")
            await self.close()
            raise

    async def close(self):
        """Close the session and cleanup resources"""
        if self.session:
            try:
                await self.session.close()
                self.session = None
                self._initialized = False
                logger.info("LLM Manager closed successfully")
            except Exception as e:
                logger.error(f"Error closing LLM Manager: {str(e)}")
                raise

    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the LLM"""
        if not self._initialized:
            raise RuntimeError("LLM Manager not initialized. Call initialize() first.")

        try:
            payload = {
                "model": self.settings.MODEL_NAME,
                "messages": messages,
                "temperature": self.settings.TEMPERATURE,
                "max_tokens": self.settings.MAX_TOKENS,
                "stream": self.settings.STREAM_ENABLED
            }

            logger.debug(f"Sending request to LLM with {len(messages)} messages")

            async with self.session.post(
                    f"{self.settings.NVIDIA_BASE_URL}/chat/completions",
                    json=payload
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    logger.error(f"LLM API Error: {response.status} - {error_text}")
                    raise Exception(f"LLM API Error: {response.status} - {error_text}")

                response_text = ""
                async for line in response.content:
                    if line:
                        try:
                            json_response = json.loads(line)
                            if content := json_response.get("choices", [{}])[0].get("delta", {}).get("content"):
                                response_text += content
                        except json.JSONDecodeError:
                            continue

                logger.debug("Response generated successfully")
                return response_text

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error processing your request."
