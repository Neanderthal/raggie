import os
import logging
import asyncio
import httpx
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from model_app.core.embedding import EmbeddingService, EmbeddingConnectionError, EmbeddingAPIError
from model_app.core.rag import store_embeddings
from model_app.db.db import get_or_create_user, get_or_create_scope, create_initial_document
from celery_app import celery_app

logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class EmbeddingTaskConfig:
    """Configuration for embedding tasks."""
    soft_time_limit: int = 300  # 5 minutes
    time_limit: int = 600       # 10 minutes
    queue: str = "embeddings_queue"


class EmbeddingTaskProcessor:
    """Processor for embedding tasks with proper error handling and logging."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.config = EmbeddingTaskConfig()

    async def process_texts_to_embeddings(
        self,
        texts: List[str],
        username: str,
        scope_name: str,
        document_name: str = "unknown",
        document_id: Optional[str] = None,
    ) -> None:
        """Process texts and generate embeddings with proper error handling."""
        logger.info(
            f"Starting embeddings task for user {username}, scope {scope_name}, document {document_name}"
        )
        if document_id:
            logger.info(f"Processing chunks for document ID: {document_id}")

        # Setup database entities
        user_id, scope_id = await self._setup_database_entities(username, scope_name)
        initial_doc_id = await self._create_initial_document(texts, user_id, scope_id)

        # Generate embeddings
        embeddings = await self._generate_embeddings_for_texts(
            texts, username, scope_name, document_name, document_id,
            user_id, scope_id, initial_doc_id
        )

        if not embeddings:
            logger.error("No embeddings generated - aborting")
            return

        # Store embeddings
        await self._store_embeddings(embeddings, initial_doc_id)

    async def _setup_database_entities(self, username: str, scope_name: str) -> tuple[int, int]:
        """Setup user and scope in database."""
        try:
            user_id = get_or_create_user(username)
            scope_id = get_or_create_scope(scope_name)
            logger.info(f"Using user ID: {user_id}, scope ID: {scope_id}")
            return user_id, scope_id
        except Exception as e:
            logger.error(f"Failed to get/create user or scope: {str(e)}")
            raise

    async def _create_initial_document(self, texts: List[str], user_id: int, scope_id: int) -> int:
        """Create initial document record."""
        try:
            full_content = "\n".join(texts)
            initial_doc_id = create_initial_document(full_content, user_id, scope_id)
            logger.info(f"Created initial document with ID: {initial_doc_id}")
            return initial_doc_id
        except Exception as e:
            logger.error(f"Failed to create initial document: {str(e)}")
            raise

    async def _generate_embeddings_for_texts(
        self,
        texts: List[str],
        username: str,
        scope_name: str,
        document_name: str,
        document_id: Optional[str],
        user_id: int,
        scope_id: int,
        initial_doc_id: int,
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for all texts."""
        embeddings = []
        
        for text in texts:
            try:
                _, embedding = await self.embedding_service.generate_embeddings(text)
                
                metadata = self._create_metadata(
                    username, scope_name, document_name, document_id,
                    user_id, scope_id, initial_doc_id
                )
                
                embeddings.append({
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata
                })
                
            except EmbeddingConnectionError:
                logger.critical("Embedding service unreachable")
                raise  # Let Celery handle retry logic
            except EmbeddingAPIError as e:
                logger.warning(f"Embedding service error: {str(e)}")
                continue  # Try next chunk
            except Exception as e:
                logger.exception("Unexpected embedding processing error")
                raise RuntimeError("Failed to process embeddings") from e

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def _create_metadata(
        self,
        username: str,
        scope_name: str,
        document_name: str,
        document_id: Optional[str],
        user_id: int,
        scope_id: int,
        initial_doc_id: int,
    ) -> Dict[str, Any]:
        """Create metadata for embedding."""
        metadata = {
            "username": username,
            "scope": scope_name,
            "scope_id": scope_id,
            "user_id": user_id,
            "document_name": document_name,
            "initial_document_id": initial_doc_id,
        }
        
        if document_id:
            metadata["document_id"] = document_id
            
        return metadata

    async def _store_embeddings(self, embeddings: List[Dict[str, Any]], initial_doc_id: int) -> None:
        """Store embeddings in database."""
        logger.info(f"Storing {len(embeddings)} embeddings in database")
        try:
            ids = await store_embeddings(embeddings, initial_document_id=initial_doc_id)
            logger.info(f"Successfully stored {len(embeddings)} document chunks in database")
            logger.debug(f"First document chunk ID: {ids[0] if ids else 'none'}")
        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}")
            raise


# Global processor instance
embedding_processor = EmbeddingTaskProcessor()


@celery_app.task(
    bind=True, 
    name="model_app.tasks.text_to_embeddings", 
    queue=embedding_processor.config.queue,
    soft_time_limit=embedding_processor.config.soft_time_limit,
    time_limit=embedding_processor.config.time_limit
)
def texts_to_embeddings(
    self,
    texts: List[str],
    username: str,
    scope_name: str,
    document_name: str = "unknown",
    document_id: Optional[str] = None,
):
    """Celery task to process texts and generate embeddings."""
    try:
        asyncio.run(
            embedding_processor.process_texts_to_embeddings(
                texts, username, scope_name, document_name, document_id
            )
        )
    except Exception as e:
        logger.error(f"Embedding task failed: {str(e)}")
        raise
