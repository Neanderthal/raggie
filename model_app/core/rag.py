import logging
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from model_app.db.db import get_rag_documents
from model_app.core.embedding import generate_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Initialize client lazily to avoid import-time errors
embedding_client = None


def get_embedding_client():
    global embedding_client
    if embedding_client is None:
        embedding_client = AsyncOpenAI(
            api_key="dummy-key", base_url=os.getenv("EMBEDDING_MODEL_URL"), timeout=30.0
        )
    return embedding_client


async def rag_query(
    query: str,
    scope: str | None = None,
    user: str | None = None,
    k: int = 5,
    similarity_threshold: float = 0.7,
) -> list[tuple[str, float]]:
    """Query RAG documents using Postgres vector store.

    Args:
        query: Search query string
        scope: Optional scope filter
        user: Optional user filter
        k: Number of results to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of tuples containing (document_content, similarity_score)
    """
    try:
        # Generate embedding for the query
        _, embedding = await generate_embeddings(query)

        # Query Postgres vector store
        raw_results = await get_rag_documents(
            scope=scope,
            user=user,
            query_embedding=embedding,
        )
        # Convert from (content, embedding, similarity) to (content, similarity)
        results = [(content, similarity) for content, _, similarity in raw_results]

        # Log results
        logger.info(f"Query: '{query}' - Found {len(results)} matching documents")
        for i, (content, similarity) in enumerate(results):
            preview = (content[:100] + "...") if len(content) > 100 else content
            logger.info(f"#{i+1} [Score: {similarity:.4f}]: {preview}")

        return results

    except ValueError as e:
        logger.warning(f"Invalid RAG parameters: {str(e)}")
        raise
    except ConnectionError as e:
        logger.warning("Embedding service unavailable")
        raise
    except Exception as e:
        logger.exception("System error in RAG")
        raise
    finally:
        pass
