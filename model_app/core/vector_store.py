"""Vector store client for RAG operations."""

import logging
from typing import Optional
from langchain_postgres import PGVector
from model_app.core.embedding import CustomLlamaEmbeddings
from model_app.core.rag_config import rag_config, RAGConfig

logger = logging.getLogger(__name__)

# Initialize PGVector client lazily
_pgvector_client = None


def get_pgvector_client(config: Optional[RAGConfig] = None) -> PGVector:
    """Get or create PGVector client with lazy initialization.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Initialized PGVector client
    """
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
