import os
import logging
import asyncio
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
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {str(e)}")
            continue

    if not embeddings:
        logger.error("No embeddings generated - aborting")
        return

    logger.info(f"Generated {len(embeddings)} embeddings")

    # Connect to pgvector
    connection_str = f"postgresql+psycopg://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'postgres')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'vector_test')}"
    logger.info(f"Connecting to database at {os.getenv('DB_HOST', 'localhost')}")
    
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
        # Convert our embeddings to LangChain document format
        documents = [
            {
                "page_content": emb["text"],
                "metadata": emb["metadata"],
                "embedding": emb["embedding"]
            } for emb in embeddings
        ]
        vectorstore.add_documents(documents)
        logger.info("Successfully stored documents in vector store")
    except Exception as e:
        logger.error(f"Failed to store documents: {str(e)}")
        raise
