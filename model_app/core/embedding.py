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
            embeddings.append(data[0]["embedding"])
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts):
        embeddings = []
        async with httpx.AsyncClient() as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    json={"content": text},
                )
                response.raise_for_status()
                data = response.json()
                # Flatten the embedding if it's nested
                embedding = data[0]["embedding"]
                if isinstance(embedding[0], list):
                    embedding = embedding[0]
                embeddings.append(embedding)
        return embeddings


async def generate_embeddings(text: str) -> Tuple[str, list[float]]:

    embedding_model = CustomLlamaEmbeddings(base_url=embedding_url)

    """Generate embeddings using the actual model with fallback"""
    text_clean = text.strip()
    if not text_clean:
        return "empty", [0.0] * 768  # Return empty embedding vector

    try:
        # Try to use the actual client to generate embeddings
        response = await embedding_model.aembed_documents([text_clean])
        return text_clean, response[0]
    except Exception as e:
        raise ConnectionError(
            f"Warning: Embedding model '{embedding_model_name}' on {embedding_url} not available. Error: {e}"
        )
        # Fallback to simple hash-based embeddings


def clean_text(text: str) -> str:
    # Trim leading/trailing whitespace
    text = text.strip().lower()
    # Replace multiple spaces/tabs with a single space (excluding newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple newlines (with optional spaces/tabs between) with a single newline
    text = re.sub(r"\s*\n\s*", "\n", text)  # Clean up spaces around newlines
    text = re.sub(r"\n+", "\n", text)  # Collapse multiple newlines into one
    return text
