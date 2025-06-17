import logging
import os
from typing import List, Tuple, Dict, Any
from uuid import uuid4
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.documents import Document
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
        embedding_url = os.getenv("EMBEDDING_MODEL_URL", "http://localhost:8000/v1")
        embedding_model = CustomLlamaEmbeddings(base_url=embedding_url)
        
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
    document_name: str | None = None,
    k: int = 5,
    similarity_threshold: float = 0.7,
) -> List[Tuple[str, float]]:
    """Query RAG documents using PGVector.

    Args:
        query: Search query string
        scope: Optional scope filter
        user: Optional user filter
        document_name: Optional document name filter
        k: Number of results to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of tuples containing (document_content, similarity_score)
    """
    try:
        # Get PGVector client
        vector_store = get_pgvector_client()
        
        # Build filter based on scope, user, and document name
        filter_dict = {}
        if scope:
            filter_dict["metadata.scope"] = scope
        if user:
            filter_dict["metadata.username"] = user
        if document_name:
            filter_dict["metadata.document_name"] = document_name
            
        # Generate embedding for the query (we'll use the embedding directly)
        _, embedding = await generate_embeddings(query)
        
        # Query PGVector with similarity search
        results_with_scores = vector_store.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k * 2,  # Request more results than needed to allow for filtering
            filter=filter_dict if filter_dict else None,
        )
        
        # Filter by similarity threshold and take top k
        filtered_results = [(doc, score) for doc, score in results_with_scores if score >= similarity_threshold]
        filtered_results = filtered_results[:k]  # Limit to k results
        
        # Convert from (Document, score) to (content, similarity)
        results = [(doc.page_content, score) for doc, score in filtered_results]

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


async def store_embeddings(embeddings_data: List[Dict[str, Any]]) -> List[str]:
    """
    Store embeddings in the PGVector database.
    
    Args:
        embeddings_data: List of dictionaries containing text, embedding, and metadata
        
    Returns:
        List of document IDs
    """
    try:
        # Get PGVector client
        vectorstore = get_pgvector_client()
        
        # Convert to LangChain documents
        documents = []
        for emb in embeddings_data:
            documents.append(
                Document(
                    page_content=emb["text"],
                    metadata=emb["metadata"],
                    embedding=emb["embedding"],
                )
            )

        # Generate UUIDs for each document
        ids = [str(uuid4()) for _ in documents]
        
        # Store documents with their IDs
        stored_ids = vectorstore.add_documents(documents=documents, ids=ids)
        
        logger.info(f"Successfully stored {len(documents)} documents in vector store")
        logger.debug(f"First document ID: {stored_ids[0] if stored_ids else 'none'}")
        
        return stored_ids
    except Exception as e:
        logger.error(f"Failed to store documents: {str(e)}")
        raise
