import os
import re
import logging
import asyncio
from typing import Tuple, List, Optional
import requests
import httpx
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load environment variables from the model_app directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


class EmbeddingConfig:
    """Configuration for embedding service."""
    
    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "tokenizer-model")
        self.base_url = os.getenv("EMBEDDING_MODEL_URL") or "http://localhost:8001/v1"
        self.default_timeout = 120.0
        self.health_check_timeout = 5.0
        self.max_retries = 3
        self.base_retry_delay = 0.5
        self.max_retry_delay = 30.0
        self.default_embedding_dimension = 768
        
        assert self.base_url is not None, "EMBEDDING_MODEL_URL must be set"


# Global configuration instance
embedding_config = EmbeddingConfig()


class EmbeddingServiceError(Exception):
    """Base exception for embedding service errors."""
    pass


class EmbeddingConnectionError(EmbeddingServiceError):
    """Raised when connection to embedding service fails."""
    pass


class EmbeddingAPIError(EmbeddingServiceError):
    """Raised when embedding API returns an error."""
    pass


class CustomLlamaEmbeddings(Embeddings):
    """Custom embedding client for LLaMA-based embedding models."""
    
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding generation."""
        embeddings = []
        for text in texts:
            res = requests.post(
                f"{self.base_url}/embeddings",
                json={"input": text, "model": "text-embedding"},
            )
            res.raise_for_status()
            data = res.json()
            embedding = self._flatten_embedding(data["data"][0]["embedding"])
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async generate embeddings with better error handling."""
        embeddings = []
        timeout = httpx.Timeout(self.timeout)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            for text in texts:
                embedding = await self._generate_single_embedding(client, text)
                embeddings.append(embedding)
        
        return embeddings

    async def _generate_single_embedding(self, client: httpx.AsyncClient, text: str) -> List[float]:
        """Generate embedding for a single text with proper error handling."""
        try:
            # Ensure text is properly encoded
            clean_text = self._prepare_text(text)
            
            response = await client.post(
                f"{self.base_url}/embeddings",
                json={"input": clean_text, "model": "text-embedding"},
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = self._flatten_embedding(data["data"][0]["embedding"])
            return embedding
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding API error (status {e.response.status_code})")
            raise EmbeddingAPIError("Embedding service returned an error") from e
        except httpx.RequestError as e:
            logger.error("Embedding service connection failed")
            raise EmbeddingConnectionError("Could not connect to embedding service") from e
        except UnicodeDecodeError as e:
            logger.error(f"Text encoding error: {e}")
            raise ValueError("Invalid text encoding") from e
        except Exception as e:
            logger.exception("Unexpected error generating embedding")
            raise RuntimeError("Failed to generate embedding") from e

    def _prepare_text(self, text: str) -> str:
        """Prepare text for embedding generation."""
        if isinstance(text, str):
            return text.encode('utf-8').decode('utf-8')
        return str(text)

    def _flatten_embedding(self, embedding: List) -> List[float]:
        """Flatten multi-dimensional embeddings if necessary."""
        if isinstance(embedding[0], (list, tuple)):
            return [item for sublist in embedding for item in sublist]
        return embedding


class EmbeddingService:
    """High-level service for embedding operations with retry logic and health checking."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or embedding_config
        self.embedding_model = CustomLlamaEmbeddings(
            base_url=self.config.base_url,
            timeout=self.config.default_timeout
        )

    async def generate_embeddings(self, text: str) -> Tuple[str, List[float]]:
        """Generate embeddings with retry logic and health checking.
        
        Args:
            text: Input text to embed
        Returns:
            Tuple of (clean_text, embedding_vector)
        Raises:
            EmbeddingConnectionError: If embedding service is unavailable
            EmbeddingAPIError: If embeddings couldn't be generated
        """
        text_clean = clean_text(text)
        if not text_clean:
            return "empty", self._get_fallback_embedding()

        for attempt in range(self.config.max_retries):
            try:
                await self._check_service_health()
                response = await self.embedding_model.aembed_documents([text_clean])
                return text_clean, response[0]
                
            except (EmbeddingConnectionError, httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(f"Attempt {attempt + 1}: Embedding service unavailable: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed after {self.config.max_retries} attempts to connect to embedding service")
                    raise EmbeddingConnectionError("Embedding service is unavailable") from e
                
                await self._wait_with_backoff(attempt)
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Unexpected error generating embedding: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed after {self.config.max_retries} attempts to generate embedding")
                    return text_clean, self._get_fallback_embedding()
                
                await self._wait_with_backoff(attempt)
        
        return text_clean, self._get_fallback_embedding()

    async def _check_service_health(self) -> None:
        """Check if embedding service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.config.health_check_timeout)) as client:
                health_url = f"{self.config.base_url.rstrip('/v1').rstrip('/')}/health"
                health = await client.get(health_url)
                if health.status_code != 200:
                    logger.warning(f"Embedding service health check failed: {health.status_code}")
        except Exception:
            logger.warning("Embedding service health check failed, trying direct embedding")

    async def _wait_with_backoff(self, attempt: int) -> None:
        """Wait with exponential backoff."""
        retry_delay = min(
            self.config.base_retry_delay * (2 ** attempt),
            self.config.max_retry_delay
        )
        await asyncio.sleep(retry_delay)

    def _get_fallback_embedding(self) -> List[float]:
        """Get fallback embedding vector."""
        return [0.0] * self.config.default_embedding_dimension


def clean_text(text: str) -> str:
    """Clean and normalize text for embedding generation."""
    # Trim leading/trailing whitespace
    text = text.strip().lower()
    # Replace multiple spaces/tabs with a single space (excluding newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple newlines (with optional spaces/tabs between) with a single newline
    text = re.sub(r"\s*\n\s*", "\n", text)  # Clean up spaces around newlines
    text = re.sub(r"\n+", "\n", text)  # Collapse multiple newlines into one
    return text


# Convenience function for backward compatibility
async def generate_embeddings(text: str) -> Tuple[str, List[float]]:
    """Generate embeddings using the default embedding service.
    
    Args:
        text: Input text to embed
    Returns:
        Tuple of (clean_text, embedding_vector)
    """
    service = EmbeddingService()
    return await service.generate_embeddings(text)
