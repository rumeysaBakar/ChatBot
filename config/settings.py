from dataclasses import dataclass
from typing import Dict


@dataclass
class Settings:
    """Application settings and configuration"""

    # Nvidia API Configuration
    NVIDIA_API_ID: str = "175dcba2-2764-40cf-b07b-5cec4e44d01c"
    NVIDIA_API_KEY: str = "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"
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    MODEL_NAME: str = "nvidia/llama-3.1-nemotron-70b-instruct"

    # Pinecone Configuration
    PINECONE_API_KEY: str = "pcsk_5TwT14_3ivRupbgDrxkunC6TmrD6M6yUeaQV6xh1NswfjEV2iTixCYwT3SW7H8HwBoToh1"  # Replace with your actual key
    PINECONE_INDEX_NAME: str = "default-index"
    PINECONE_ENVIRONMENT: str = "gcp-starter"

    # Vector Store Configuration
    VECTOR_DIMENSION: int = 1536

    # Model Parameters
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.7
    SUMMARY_TEMPERATURE: float = 0.3
    STREAM_ENABLED: bool = True

    # Memory Management
    MAX_CACHE_ITEMS: int = 1000
    CACHE_CLEANUP_THRESHOLD: int = 500

    # Performance Tuning
    MAX_PARALLEL_REQUESTS: int = 4
    REQUEST_TIMEOUT: int = 30

    def __post_init__(self):
        """Validate settings after initialization"""
        required_fields = [
            'NVIDIA_API_KEY',
            'PINECONE_API_KEY',
            'PINECONE_INDEX_NAME',
            'PINECONE_ENVIRONMENT'
        ]

        for field in required_fields:
            if not getattr(self, field, None):
                raise ValueError(f"Missing required configuration: {field}")

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "NVIDIA-API-ID": self.NVIDIA_API_ID
        }
