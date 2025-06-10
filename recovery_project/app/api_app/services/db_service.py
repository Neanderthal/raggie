"""
Database service for RAG operations.

This module provides a service for interacting with the database for RAG operations.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor

from api_app.config import settings

logger = logging.getLogger(__name__)

class DBService:
    """
    Service for database operations.
    
    This class provides methods to interact with the database for RAG operations.
    """
    
    def __init__(self):
        """Initialize the database service with connection parameters."""
        self.db_params = {
            "dbname": settings.DB_NAME,
            "user": settings.DB_USER,
            "password": settings.DB_PASSWORD,
            "host": settings.DB_HOST,
            "port": settings.DB_PORT,
        }
        logger.info(f"Initialized DB service with host: {settings.DB_HOST}")
        
    def get_connection(self):
        """Get a database connection."""
        try:
            conn = psycopg2.connect(**self.db_params, cursor_factory=RealDictCursor)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
            
    async def get_rag_documents(
        self,
        query_embedding: List[float],
        scope: Optional[str] = None,
        user: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """
        Get relevant documents for RAG.
        
        Args:
            query_embedding: The query embedding vector
            scope: Optional scope to filter documents by
            user: Optional username to filter documents by
            limit: Maximum number of documents to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            A list of tuples containing (document_content, similarity_score)
        """
        try:
            # Build the query with optional filters
            query = """
                SELECT 
                    content,
                    1 - (embedding <=> %s) as similarity
                FROM documents
                WHERE 1 = 1
            """
            params: List[Any] = [query_embedding]
            
            # Add optional filters
            if scope:
                query += " AND scope = %s"
                params.append(scope)
                
            if user:
                query += " AND username = %s"
                params.append(user)
                
            # Add similarity threshold and order by
            query += """
                AND 1 - (embedding <=> %s) > %s
                ORDER BY similarity DESC
                LIMIT %s
            """
            params.extend([query_embedding, similarity_threshold, limit])
            
            # Execute in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_rag_query, query, params
            )
            
            return result
        except Exception as e:
            logger.error(f"Error getting RAG documents: {str(e)}")
            return []
            
    def _execute_rag_query(self, query: str, params: List[Any]) -> List[Tuple[str, float]]:
        """Execute a RAG query synchronously."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()
                
                # Convert to list of tuples (content, similarity)
                return [(row["content"], row["similarity"]) for row in results]
                
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            document_id: The document ID
            
        Returns:
            The document as a dictionary, or None if not found
        """
        try:
            query = "SELECT * FROM documents WHERE id = %s"
            
            # Execute in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_get_document, query, [document_id]
            )
            
            return result
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            return None
            
    def _execute_get_document(self, query: str, params: List[Any]) -> Optional[Dict[str, Any]]:
        """Execute a get document query synchronously."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                
                if result:
                    return dict(result)
                return None

# Create a global database service instance
db_service = DBService()
