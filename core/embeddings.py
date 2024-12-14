import aiohttp
import numpy as np
from typing import List
from config.settings import Settings
from utils.async_utils import AsyncSessionManager


class NvidiaEmbeddings(AsyncSessionManager):
    def __init__(self, settings: Settings):
        super().__init__(settings)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        async with self._get_session() as session:
            embeddings = []

            for text in texts:
                async with session.post(
                        f"{self.settings.NVIDIA_BASE_URL}/embeddings",
                        json={
                            "model": self.settings.MODEL_NAME,
                            "input": text
                        }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        embeddings.append(result["data"][0]["embedding"])
                    else:
                        raise Exception(f"Embedding error: {await response.text()}")

            return embeddings