"""Exception classes for RAG operations."""


class RAGServiceError(Exception):
    """Base exception for RAG service errors."""
    pass


class RAGQueryError(RAGServiceError):
    """Raised when RAG query fails."""
    pass


class RAGStorageError(RAGServiceError):
    """Raised when RAG storage operation fails."""
    pass
