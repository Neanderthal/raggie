import os
import logging
import asyncio
import httpx
from typing import List, Optional
from model_app.core.embedding import generate_embeddings
from model_app.core.rag import store_embeddings
from model_app.db.db import get_or_create_user, get_or_create_scope
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
    document_id: Optional[str] = None,
):
    logger.info(
        f"Starting embeddings task for user {username}, scope {scope_name}, document {document_name}"
    )
    if document_id:
        logger.info(f"Processing chunks for document ID: {document_id}")

    # Ensure user and scope exist in the database
    try:
        # These functions are not coroutines, so we don't need asyncio.run
        user_id = get_or_create_user(username)
        scope_id = get_or_create_scope(scope_name)
        logger.info(f"Using user ID: {user_id}, scope ID: {scope_id}")
    except Exception as e:
        logger.error(f"Failed to get/create user or scope: {str(e)}")
        raise

    # Generate embeddings for each text chunk
    embeddings = []
    for text in texts:
        try:
            _, embedding = asyncio.run(generate_embeddings(text))

            # Create metadata with document ID if available
            metadata = {
                "username": username,
                "scope": scope_name,
                "scope_id": scope_id,
                "user_id": user_id,
                "document_name": document_name,
            }

            # Add document_id to metadata if provided
            if document_id:
                metadata["document_id"] = document_id

            embeddings.append(
                {"text": text, "embedding": embedding, "metadata": metadata}
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
    logger.info(f"Storing {len(embeddings)} embeddings in database")
    try:
        ids = asyncio.run(store_embeddings(embeddings))
        logger.info(
            f"Successfully stored {len(embeddings)} document chunks in database"
        )
        logger.debug(f"First document chunk ID: {ids[0] if ids else 'none'}")
    except Exception as e:
        logger.error(f"Failed to store documents: {str(e)}")
        raise
