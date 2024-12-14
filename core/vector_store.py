import logging
from typing import List, Tuple, Dict, Any
from pinecone import Pinecone, PodSpec
from config.settings import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, settings: Settings, embeddings):
        self.settings = settings
        self.embeddings = embeddings
        self.pc = None
        self.index = None

    async def initialize(self):
        """Initialize the vector store with better error handling"""
        try:
            logger.info("Initializing Pinecone connection...")
            self.pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)

            # List existing indexes
            existing_indexes = self.pc.list_indexes().names()
            logger.info(f"Found existing indexes: {existing_indexes}")

            if self.settings.PINECONE_INDEX_NAME in existing_indexes:
                logger.info(f"Using existing index: {self.settings.PINECONE_INDEX_NAME}")
                self.index = self.pc.Index(self.settings.PINECONE_INDEX_NAME)
            else:
                if not existing_indexes:
                    logger.info(f"Creating new index: {self.settings.PINECONE_INDEX_NAME}")
                    try:
                        self.pc.create_index(
                            name=self.settings.PINECONE_INDEX_NAME,
                            dimension=self.settings.VECTOR_DIMENSION,
                            metric="cosine",
                            spec=PodSpec(environment=self.settings.PINECONE_ENVIRONMENT)
                        )
                        self.index = self.pc.Index(self.settings.PINECONE_INDEX_NAME)
                    except Exception as e:
                        logger.error(f"Error creating new index: {str(e)}")
                        raise
                else:
                    # Use first available index
                    index_name = existing_indexes[0]
                    logger.info(f"Using existing index: {index_name}")
                    self.index = self.pc.Index(index_name)
                    self.settings.PINECONE_INDEX_NAME = index_name

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise