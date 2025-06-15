import os
import logging
import asyncio
import httpx
from typing import List, Dict, Any
from langchain_postgres import PGVector
from model_app.core.embedding import generate_embeddings, CustomLlamaEmbeddings
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
    model_name: str = embedding_model_name,
):
    logger.info(f"Starting embeddings task for user {username}, scope {scope_name}")
    
    # Generate embeddings for each text chunk
    embeddings = []
    for text in texts:
        try:
            _, embedding = asyncio.run(generate_embeddings(text))
            embeddings.append({
                "text": text,
                "embedding": embedding,
                "metadata": {
                    "username": username,
                    "scope": scope_name
                }
            })
        except ConnectionError as e:
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

    # Connect to pgvector
    connection_str = f"postgresql+psycopg://{os.getenv('DB_USER', 'pgvector')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'db')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'pgvector_rag')}"
    logger.info(f"Connecting to database at {os.getenv('DB_HOST', 'db')}")
    
    embedding_model = CustomLlamaEmbeddings(base_url=embedding_url)
    vectorstore = PGVector(
        collection_name="rag_docs",
        connection=connection_str,
        embeddings=embedding_model,
        use_jsonb=True,
    )

    # Store embeddings
    logger.info(f"Storing {len(embeddings)} embeddings in vector store")
    try:
        # Convert to LangChain documents and store with unique IDs
        from langchain_core.documents import Document
        from uuid import uuid4
        
        documents = []
        for idx, emb in enumerate(embeddings):
            documents.append(
                Document(
                    page_content=emb["text"],
                    metadata=emb["metadata"],
                    embedding=emb["embedding"]
                )
            )
        
        # Generate UUIDs for each document
        ids = [str(uuid4()) for _ in documents]
        ids = vectorstore.add_documents(documents=documents, ids=ids)
        
        logger.info(f"Successfully stored {len(documents)} documents in vector store")
        logger.debug(f"First document ID: {ids[0] if ids else 'none'}")
    except Exception as e:
        logger.error(f"Failed to store documents: {str(e)}")
        raise
