"""Facade module for RAG operations.

This module provides a simple interface to the RAG functionality
by re-exporting components from specialized modules.
"""

import logging
import os
from typing import List, Tuple, Dict, Any, Optional
from uuid import uuid4
from dataclasses import dataclass
from langchain_postgres import PGVector
from langchain_core.documents import Document
from model_app.core.embedding import CustomLlamaEmbeddings

# Set up logger
logger = logging.getLogger(__name__)

# Re-export configuration
from model_app.core.rag_config import RAGConfig as ConfigRAGConfig, rag_config

# Re-export exceptions
from model_app.core.rag_exceptions import (
    RAGServiceError as ExceptionRAGServiceError,
    RAGQueryError as ExceptionRAGQueryError, 
    RAGStorageError as ExceptionRAGStorageError
)

# Re-export vector store client
from model_app.core.vector_store import get_pgvector_client

# Re-export RAG service
from model_app.core.rag_service import (
    RAGService as ServiceRAGService, 
    rag_service,
    rag_query,
    store_embeddings
)

# Re-export text chunker
from model_app.core.text_chunker_service import (
    TextChunker as ServiceTextChunker,
    text_chunker,
    chunk_text_legacy
)

# Type aliases for backward compatibility
RAGConfig = ConfigRAGConfig
RAGServiceError = ExceptionRAGServiceError
RAGQueryError = ExceptionRAGQueryError
RAGStorageError = ExceptionRAGStorageError
RAGService = ServiceRAGService
TextChunker = ServiceTextChunker

# For backward compatibility, keep the original class definition
# The duplicate class definitions have been removed
# All functionality is now imported from the specialized modules
