import os
import re
from typing import Tuple
import requests
import httpx
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import os

# Load environment variables from the model_app directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "tokenizer-model")
embedding_url = os.getenv("EMBEDDING_MODEL_URL")


class CustomLlamaEmbeddings(Embeddings):
    def __init__(self, base_url: str):
        self.base_url = base_url  # e.g. "http://tokenizer_model:8000"

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            res = requests.post(
                f"{self.base_url}/embeddings",
                json={"content": text},
            )
            res.raise_for_status()
            data = res.json()
            embedding = data[0]["embedding"]
            if isinstance(embedding[0], (list, tuple)):  # Flatten if multi-dimensional
                embedding = [item for sublist in embedding for item in sublist]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts):
        """Async generate embeddings with better error handling."""
        embeddings = []
        timeout = httpx.Timeout(30.0)  # 30 second timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            for text in texts:
                try:
                    response = await client.post(
                        f"{self.base_url}/embeddings",
                        json={"content": text},
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    data = response.json()
                    embedding = data[0]["embedding"]
                    if isinstance(embedding[0], (list, tuple)):  
                        embedding = [item for sublist in embedding for item in sublist]
                    embeddings.append(embedding)
                except httpx.HTTPStatusError as e:
                    logger.error(f"Embedding API error (status {e.response.status_code})")
                    raise ConnectionError("Embedding service returned an error") from e
                except httpx.RequestError as e:
                    logger.error("Embedding service connection failed")
                    raise ConnectionError("Could not connect to embedding service") from e
                except Exception as e:
                    logger.exception("Unexpected error generating embedding")
                    raise RuntimeError("Failed to generate embedding") from e
        return embeddings


async def generate_embeddings(text: str) -> Tuple[str, list[float]]:
    """Generate embeddings with retry logic and health checking."""
    embedding_model = CustomLlamaEmbeddings(base_url=embedding_url)
    text_clean = text.strip()
    if not text_clean:
        return "empty", [0.0] * 768

    max_retries = 3
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Check tokenizer health first
            async with httpx.AsyncClient() as client:
                health = await client.get(f"{embedding_url.rstrip('/v1')}/health")
                if health.status_code != 200:
                    raise ConnectionError(f"Tokenizer health check failed: {health.status_code}")
                
                response = await embedding_model.aembed_documents([text_clean])
                return text_clean, response[0]
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise ConnectionError(
                    f"Failed after {max_retries} attempts to generate embedding: {str(e)}"
                )
            await asyncio.sleep(retry_delay * (attempt + 1))


def clean_text(text: str) -> str:
    # Trim leading/trailing whitespace
    text = text.strip().lower()
    # Replace multiple spaces/tabs with a single space (excluding newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple newlines (with optional spaces/tabs between) with a single newline
    text = re.sub(r"\s*\n\s*", "\n", text)  # Clean up spaces around newlines
    text = re.sub(r"\n+", "\n", text)  # Collapse multiple newlines into one
    return text
