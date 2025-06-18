import logging
import os
from typing import List, Tuple, Dict, Any, Optional
from uuid import uuid4
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.documents import Document
from model_app.core.embedding import CustomLlamaEmbeddings
from model_app.db.db import store_vector_document_links

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


@dataclass
class RAGConfig:
    """Configuration for RAG operations."""
    
    def __init__(self):
        self.collection_name = "rag_docs"
        self.default_k = 5
        self.default_similarity_threshold = 0.2
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.use_jsonb = True
        
        # Database connection
        self.db_user = os.getenv('DB_USER', 'pgvector')
        self.db_password = os.getenv('DB_PASSWORD', 'password')
        self.db_host = os.getenv('DB_HOST', 'db')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'pgvector_rag')
        
        # Embedding service
        self.embedding_url = os.getenv("EMBEDDING_MODEL_URL", "http://localhost:8001/v1")

    @property
    def connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql+psycopg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


class RAGServiceError(Exception):
    """Base exception for RAG service errors."""
    pass


class RAGQueryError(RAGServiceError):
    """Raised when RAG query fails."""
    pass


class RAGStorageError(RAGServiceError):
    """Raised when RAG storage operation fails."""
    pass


# Global configuration instance
rag_config = RAGConfig()

# Initialize PGVector client lazily
_pgvector_client = None


def get_pgvector_client(config: Optional[RAGConfig] = None) -> PGVector:
    """Get or create PGVector client with lazy initialization."""
    global _pgvector_client
    if _pgvector_client is None:
        config = config or rag_config
        
        # Initialize embedding model
        embedding_model = CustomLlamaEmbeddings(base_url=config.embedding_url)

        # Create PGVector client
        _pgvector_client = PGVector(
            collection_name=config.collection_name,
            connection=config.connection_string,
            embeddings=embedding_model,
            use_jsonb=config.use_jsonb,
        )
    return _pgvector_client


class RAGService:
    """Service for RAG (Retrieval-Augmented Generation) operations."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or rag_config
        self._vector_store = None

    @property
    def vector_store(self) -> PGVector:
        """Get vector store with lazy initialization."""
        if self._vector_store is None:
            self._vector_store = get_pgvector_client(self.config)
        return self._vector_store

    async def query_documents(
        self,
        query: str,
        scope: Optional[str] = None,
        user: Optional[str] = None,
        document_name: Optional[str] = None,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
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
            
        Raises:
            RAGQueryError: If query fails
        """
        k = k or self.config.default_k
        similarity_threshold = similarity_threshold or self.config.default_similarity_threshold
        
        try:
            # Build filter using PGVector's native filtering syntax
            filter_dict = self._build_filter(scope, user, document_name)

            # Use PGVector's native similarity search with filtering
            # TODO: make it async by using asimilarity_search_with_score
            if filter_dict:
                results_with_scores = self.vector_store.similarity_search_with_score(
                    query=query, k=k, filter=filter_dict
                )
            else:
                results_with_scores = self.vector_store.similarity_search_with_score(
                    query=query, k=k
                )

            # Filter by similarity threshold and convert results
            results = self._process_query_results(results_with_scores, similarity_threshold)
            
            # Log results
            self._log_query_results(query, results)

            return results

        except ValueError as e:
            logger.warning(f"Invalid RAG parameters: {str(e)}")
            raise RAGQueryError(f"Invalid query parameters: {str(e)}") from e
        except ConnectionError as e:
            logger.warning("Embedding service unavailable")
            raise RAGQueryError("Embedding service unavailable") from e
        except Exception as e:
            logger.exception(f"System error in RAG query: {str(e)}")
            raise RAGQueryError(f"Query failed: {str(e)}") from e

    def _build_filter(
        self, 
        scope: Optional[str], 
        user: Optional[str], 
        document_name: Optional[str]
    ) -> Dict[str, Any]:
        """Build filter dictionary for PGVector query."""
        filter_dict = {}
        if scope:
            filter_dict["scope"] = {"$eq": scope}
        if user:
            filter_dict["username"] = {"$eq": user}
        if document_name:
            filter_dict["document_name"] = {"$eq": document_name}
        return filter_dict

    def _process_query_results(
        self, 
        results_with_scores: List[Tuple[Document, float]], 
        similarity_threshold: float
    ) -> List[Tuple[str, float]]:
        """Process and filter query results."""
        # Filter by similarity threshold
        filtered_results = [
            (doc, score)
            for doc, score in results_with_scores
            if score >= similarity_threshold
        ]

        # Convert from (Document, score) to (content, similarity)
        return [(doc.page_content, score) for doc, score in filtered_results]

    def _log_query_results(self, query: str, results: List[Tuple[str, float]]) -> None:
        """Log query results for debugging."""
        logger.info(f"Query: '{query}' - Found {len(results)} matching documents")
        for i, (content, similarity) in enumerate(results):
            preview = (content[:100] + "...") if len(content) > 100 else content
            logger.info(f"#{i+1} [Score: {similarity:.4f}]: {preview}")

    async def store_embeddings(
        self, 
        embeddings_data: List[Dict[str, Any]], 
        initial_document_id: Optional[int] = None
    ) -> List[str]:
        """Store embeddings in the PGVector database.

        Args:
            embeddings_data: List of dictionaries containing text, embedding, and metadata
            initial_document_id: ID of the initial document these chunks came from

        Returns:
            List of document IDs
            
        Raises:
            RAGStorageError: If storage fails
        """
        try:
            # Convert to LangChain documents
            documents = self._prepare_documents(embeddings_data)

            # Generate UUIDs for each document
            ids = [str(uuid4()) for _ in documents]

            # Store documents with their IDs
            stored_ids = self.vector_store.add_documents(documents=documents, ids=ids)

            # If initial_document_id is provided, store the relationship
            if initial_document_id:
                store_vector_document_links(initial_document_id, stored_ids)

            logger.info(f"Successfully stored {len(documents)} documents in vector store")
            logger.debug(f"First document ID: {stored_ids[0] if stored_ids else 'none'}")

            return stored_ids
            
        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}")
            raise RAGStorageError(f"Storage failed: {str(e)}") from e

    def _prepare_documents(self, embeddings_data: List[Dict[str, Any]]) -> List[Document]:
        """Convert embeddings data to LangChain documents."""
        documents = []
        for emb in embeddings_data:
            documents.append(
                Document(
                    page_content=emb["text"],
                    metadata=emb["metadata"],
                )
            )
        return documents


# Global service instance
rag_service = RAGService()


# Convenience functions for backward compatibility
async def rag_query(
    query: str,
    scope: str | None = None,
    user: str | None = None,
    document_name: str | None = None,
    k: int = 5,
    similarity_threshold: float = 0.2,
) -> List[Tuple[str, float]]:
    """Query RAG documents using the default RAG service."""
    return await rag_service.query_documents(
        query, scope, user, document_name, k, similarity_threshold
    )


async def store_embeddings(
    embeddings_data: List[Dict[str, Any]], 
    initial_document_id: Optional[int] = None
) -> List[str]:
    """Store embeddings using the default RAG service."""
    return await rag_service.store_embeddings(embeddings_data, initial_document_id)


class TextChunker:
    """Service for chunking text into semantic pieces."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or rag_config

    def chunk_text(self, text: str) -> List[str]:
        """Split text into semantic chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        chunks = []
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        current_chunk = ""
        for sentence in sentences:
            # Use configurable chunk size
            if len(current_chunk) + len(sentence) < self.config.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_text_with_overlap(self, text: str) -> List[str]:
        """Split text into chunks with overlap for better context preservation.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks with overlap
        """
        if not text.strip():
            return []

        # Simple implementation - can be enhanced with more sophisticated chunking
        words = text.split()
        chunks = []
        
        words_per_chunk = self.config.chunk_size // 5  # Rough estimate
        overlap_words = self.config.chunk_overlap // 5
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
                
        return chunks


# Global chunker instance
text_chunker = TextChunker()


# Convenience function for backward compatibility
def chunk_text(text: str) -> List[str]:
    """Split text into semantic chunks using the default chunker."""
    return text_chunker.chunk_text(text)
