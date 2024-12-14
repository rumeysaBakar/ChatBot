import aiohttp
import asyncio
from contextlib import asynccontextmanager
from config.settings import Settings


class AsyncSessionManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._session = None
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def _get_session(self):
        if self._session is None:
            async with self._lock:
                if self._session is None:
                    self._session = aiohttp.ClientSession(
                        headers=self.settings.headers,
                        timeout=aiohttp.ClientTimeout(total=self.settings.REQUEST_TIMEOUT)
                    )
        try:
            yield self._session
        except Exception as e:
            await self.close()
            raise e

    async def close(self):
        """Close the session and cleanup resources"""
        if self._session:
            await self._session.close()
            self._session = None