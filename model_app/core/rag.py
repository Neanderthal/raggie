import logging
import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_postgres import PGVector
from model_app.core.embedding import generate_embeddings, CustomLlamaEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Initialize PGVector client lazily
_pgvector_client = None

def get_pgvector_client():
    global _pgvector_client
    if _pgvector_client is None:
        # Create connection string
        connection_str = f"postgresql+psycopg://{os.getenv('DB_USER', 'pgvector')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'db')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'pgvector_rag')}"
        
        # Initialize embedding model
        embedding_model = CustomLlamaEmbeddings(base_url=os.getenv("EMBEDDING_MODEL_URL"))
        
        # Create PGVector client
        _pgvector_client = PGVector(
            collection_name="rag_docs",
            connection=connection_str,
            embeddings=embedding_model,
            use_jsonb=True,
        )
    return _pgvector_client


async def rag_query(
    query: str,
    scope: str | None = None,
    user: str | None = None,
    k: int = 5,
    similarity_threshold: float = 0.7,
) -> List[Tuple[str, float]]:
    """Query RAG documents using PGVector.

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
        # Get PGVector client
        vector_store = get_pgvector_client()
        
        # Build filter based on scope and user
        filter_dict = {}
        if scope:
            filter_dict["metadata.scope"] = scope
        if user:
            filter_dict["metadata.username"] = user
            
        # Generate embedding for the query (we'll use the embedding directly)
        _, embedding = await generate_embeddings(query)
        
        # Query PGVector with similarity search
        results_with_scores = vector_store.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter_dict if filter_dict else None,
            score_threshold=similarity_threshold
        )
        
        # Convert from (Document, score) to (content, similarity)
        results = [(doc.page_content, score) for doc, score in results_with_scores]

        # Log results
        logger.info(f"Query: '{query}' - Found {len(results)} matching documents")
        for i, (content, similarity) in enumerate(results):
            preview = (content[:100] + "...") if len(content) > 100 else content
            logger.info(f"#{i+1} [Score: {similarity:.4f}]: {preview}")

        return results

    except ValueError as e:
        logger.warning(f"Invalid RAG parameters: {str(e)}")
        raise
    except ConnectionError:
        logger.warning("Embedding service unavailable")
        raise
    except Exception as e:
        logger.exception(f"System error in RAG: {str(e)}")
        raise


def chunk_text(text: str) -> list[str]:
    """Split text into semantic chunks"""
    if not text.strip():
        return []

    chunks = []
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    current_chunk = ""
    for sentence in sentences:
        # Aim for chunks of roughly 200-500 characters
        if len(current_chunk) + len(sentence) < 500:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
