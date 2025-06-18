"""Configuration module for RAG operations."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
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


# Global configuration instance
rag_config = RAGConfig()
