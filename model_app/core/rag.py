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

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

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
) -> list[tuple[str, list[float], float]] | str:
    """Store RAG conversation with optional scope and user filtering"""
    try:
        # Get embeddings from containerized service
        _, embedding = await generate_embeddings(query)
        results = await get_rag_documents(scope, user, embedding)

        # Filter out documents with duplicate similarity scores (likely duplicates)
        filtered_results = []
        seen_similarities = set()
        seen_contents = set()

        for doc in results:
            content = doc[0]
            similarity = doc[2]  # The similarity score is now the third element

            # Create a content hash for quick comparison (first 100 chars should be enough to identify duplicates)
            content_hash = content[:100]

            # Skip if we've seen this similarity score or content before
            if similarity in seen_similarities or content_hash in seen_contents:
                logger.info(
                    f"Filtering out duplicate document with similarity: {similarity:.4f}"
                )
                continue

            seen_similarities.add(similarity)
            seen_contents.add(content_hash)
            filtered_results.append(doc)

        # Log the filtered documents
        logger.info(
            f"Query: '{query}' - Found {len(results)} documents, filtered to {len(filtered_results)}"
        )
        for i, doc in enumerate(filtered_results):
            # Extract content and similarity score
            content = doc[0]
            similarity = doc[2]  # The similarity score is now the third element

            # Log a preview of the document with similarity score
            preview = content[:100] + "..." if len(content) > 100 else content
            logger.info(f"Document {i+1} (similarity: {similarity:.4f}): {preview}")
            logger.debug(f"Document {i+1} FULL CONTENT: {content}")

        return filtered_results

    except Exception as e:
        logging.error(f"Error in RAG query: {str(e)}")
        return f"Error processing your request: {str(e)}"
    finally:
        pass
