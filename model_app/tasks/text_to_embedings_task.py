import os
import logging
import asyncio
import httpx
from typing import List
from model_app.core.embedding import generate_embeddings
from model_app.core.rag import store_embeddings
from celery_app import celery_app

logger = logging.getLogger(__name__)

# Configuration
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "tokenizer-model")
embedding_url = os.getenv("EMBEDDING_MODEL_URL", "http://localhost:8001/v1")


@celery_app.task(
    bind=True, name="model_app.tasks.text_to_embeddings", queue="embeddings_queue"
)
def texts_to_embeddings(
    self,
    texts: List[str],
    username: str,
    scope_name: str,
    document_name: str = "unknown",
    model_name: str = embedding_model_name,
):
    logger.info(f"Starting embeddings task for user {username}, scope {scope_name}, document {document_name}")

    # Generate embeddings for each text chunk
    embeddings = []
    for text in texts:
        try:
            _, embedding = asyncio.run(generate_embeddings(text))
            embeddings.append(
                {
                    "text": text,
                    "embedding": embedding,
                    "metadata": {
                        "username": username, 
                        "scope": scope_name,
                        "document_name": document_name
                    },
                }
            )
        except ConnectionError:
            # Bubbled-up + Non-Recoverable
            logger.critical("Embedding service unreachable")
            raise  # Let Celery handle retry logic
        except httpx.HTTPStatusError as e:
            # Bubbled-up + Possibly Recoverable
            logger.warning(
                f"Embedding service error ({e.response.status_code}): {str(e)}"
            )
            continue  # Try next chunk
        except Exception as e:
            # Bubbled-up + Non-Recoverable
            logger.exception("Unexpected embedding processing error")
            raise RuntimeError("Failed to process embeddings") from e

    if not embeddings:
        logger.error("No embeddings generated - aborting")
        return

    logger.info(f"Generated {len(embeddings)} embeddings")

    # Store embeddings using the function from rag.py
    logger.info(f"Storing {len(embeddings)} embeddings in vector store")
    try:
        ids = asyncio.run(store_embeddings(embeddings))
        logger.info(f"Successfully stored {len(embeddings)} documents in vector store")
        logger.debug(f"First document ID: {ids[0] if ids else 'none'}")
    except Exception as e:
        logger.error(f"Failed to store documents: {str(e)}")
        raise
