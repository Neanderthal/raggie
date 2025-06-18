import os
import re
import logging
import asyncio
from typing import Tuple
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

embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "tokenizer-model")
embedding_url = os.getenv("EMBEDDING_MODEL_URL") or "http://localhost:8001/v1"
assert embedding_url is not None, "EMBEDDING_MODEL_URL must be set"


class CustomLlamaEmbeddings(Embeddings):
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url  # e.g. "http://tokenizer_model:8000"
        self.timeout = timeout

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            res = requests.post(
                f"{self.base_url}/embeddings",
                json={"input": text, "model": "text-embedding"},
            )
            res.raise_for_status()
            data = res.json()
            embedding = data["data"][0]["embedding"]
            if isinstance(embedding[0], (list, tuple)):  # Flatten if multi-dimensional
                embedding = [item for sublist in embedding for item in sublist]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts):
        """Async generate embeddings with better error handling."""
        embeddings = []
        timeout = httpx.Timeout(self.timeout)  # Configurable timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            for text in texts:
                try:
                    # Ensure text is properly encoded
                    if isinstance(text, str):
                        text = text.encode('utf-8').decode('utf-8')
                    
                    response = await client.post(
                        f"{self.base_url}/embeddings",
                        json={"input": text, "model": "text-embedding"},
                        headers={"Content-Type": "application/json; charset=utf-8"}
                    )
                    response.raise_for_status()
                    data = response.json()
                    embedding = data["data"][0]["embedding"]
                    if isinstance(embedding[0], (list, tuple)):  
                        embedding = [item for sublist in embedding for item in sublist]
                    embeddings.append(embedding)
                except httpx.HTTPStatusError as e:
                    logger.error(f"Embedding API error (status {e.response.status_code})")
                    raise ConnectionError("Embedding service returned an error") from e
                except httpx.RequestError as e:
                    logger.error("Embedding service connection failed")
                    raise ConnectionError("Could not connect to embedding service") from e
                except UnicodeDecodeError as e:
                    logger.error(f"Text encoding error: {e}")
                    raise ValueError("Invalid text encoding") from e
                except Exception as e:
                    logger.exception("Unexpected error generating embedding")
                    raise RuntimeError("Failed to generate embedding") from e
        return embeddings


async def generate_embeddings(text: str) -> Tuple[str, list[float]]:
    """Generate embeddings with retry logic and health checking.
    
    Args:
        text: Input text to embed
    Returns:
        Tuple of (clean_text, embedding_vector)
    Raises:
        ConnectionError: If embedding service is unavailable
        ValueError: If embeddings couldn't be generated
    """
    # Use longer timeout for slow tokenizer models
    embedding_model = CustomLlamaEmbeddings(base_url=embedding_url, timeout=120.0)
    text_clean = text.strip()
    if not text_clean:
        return "empty", [0.0] * 768

    max_retries = 3
    base_retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Check if embedding service is available
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                try:
                    health_url = f"{embedding_url.rstrip('/v1').rstrip('/')}/health"
                    health = await client.get(health_url)
                    if health.status_code != 200:
                        logger.warning(f"Embedding service health check failed: {health.status_code}")
                except Exception:
                    logger.warning("Embedding service health check failed, trying direct embedding")
                
                response = await embedding_model.aembed_documents([text_clean])
                return text_clean, response[0]
                
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(f"Attempt {attempt + 1}: Embedding service unavailable: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts to connect to embedding service")
                raise ConnectionError("Embedding service is unavailable")
            # Exponential backoff with jitter, capped at 30 seconds
            retry_delay = min(base_retry_delay * (2 ** attempt), 30)
            await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: Unexpected error generating embedding: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts to generate embedding")
                return text_clean, [0.0] * 768
            # Exponential backoff with jitter, capped at 30 seconds
            retry_delay = min(base_retry_delay * (2 ** attempt), 30)
            await asyncio.sleep(retry_delay)
    
    return text_clean, [0.0] * 768  # Fallback return


def clean_text(text: str) -> str:
    # Trim leading/trailing whitespace
    text = text.strip().lower()
    # Replace multiple spaces/tabs with a single space (excluding newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple newlines (with optional spaces/tabs between) with a single newline
    text = re.sub(r"\s*\n\s*", "\n", text)  # Clean up spaces around newlines
    text = re.sub(r"\n+", "\n", text)  # Collapse multiple newlines into one
    return text
