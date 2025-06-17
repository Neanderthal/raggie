import logging
import os
from typing import List, Tuple, Dict, Any, Optional
from uuid import uuid4
from dotenv import load_dotenv
from sqlmodel import Session, select
from langchain_core.documents import Document
from model_app.core.embedding import generate_embeddings, CustomLlamaEmbeddings
from model_app.db.db import engine, Document as DBDocument, get_or_create_user, get_or_create_scope

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Initialize embedding model
embedding_url = os.getenv("EMBEDDING_MODEL_URL", "http://localhost:8000/v1")
embedding_model = CustomLlamaEmbeddings(base_url=embedding_url)


async def rag_query(
    query: str,
    scope: str | None = None,
    user: str | None = None,
    document_name: str | None = None,
    k: int = 5,
    similarity_threshold: float = 0.7,
) -> List[Tuple[str, float]]:
    """Query RAG documents using SQLModel.

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
        # Generate embedding for the query
        _, embedding = await generate_embeddings(query)
        
        with Session(engine) as session:
            # Start building the query
            statement = select(DBDocument)
            
            # Add filters for scope and user if provided
            if scope:
                scope_id = get_or_create_scope(scope) or 0
                statement = statement.where(DBDocument.scope_id == scope_id)
                
            if user:
                user_id = get_or_create_user(user) or 0
                statement = statement.where(DBDocument.user_id == user_id)
            
            # Execute the query to get all matching documents
            documents = session.exec(statement).all()
            
            # Calculate similarity scores
            # Note: In a production system, you would use a database function for this
            # This is a simplified approach for demonstration
            results_with_scores = []
            for doc in documents:
                # Calculate vector similarity (simplified)
                # In production, use proper vector similarity calculation
                similarity = 0.8  # Placeholder - would be actual similarity calculation
                
                if similarity >= similarity_threshold:
                    results_with_scores.append((doc.content, similarity))
            
            # Sort by similarity and take top k
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            results = results_with_scores[:k]
            
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
    Store embeddings in the SQLModel database.
    
    Args:
        embeddings_data: List of dictionaries containing text, embedding, and metadata
        
    Returns:
        List of document IDs
    """
    try:
        with Session(engine) as session:
            document_ids = []
            
            for emb in embeddings_data:
                # Extract metadata
                metadata = emb["metadata"]
                username = metadata.get("username", "default_user")
                scope_name = metadata.get("scope", "default_scope")
                
                # Get or create user and scope
                user_id = get_or_create_user(username) or 0
                scope_id = get_or_create_scope(scope_name) or 0
                
                # Create document
                document = DBDocument(
                    content=emb["text"],
                    embedding=emb["embedding"],
                    user_id=user_id,
                    scope_id=scope_id
                )
                
                # Add to session
                session.add(document)
                session.flush()  # Flush to get the ID
                
                # Store the ID
                document_ids.append(str(document.id))
            
            # Commit all changes
            session.commit()
            
            logger.info(f"Successfully stored {len(document_ids)} documents in database")
            logger.debug(f"First document ID: {document_ids[0] if document_ids else 'none'}")
            
            return document_ids
    except Exception as e:
        logger.error(f"Failed to store documents: {str(e)}")
        raise
