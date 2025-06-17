"""
Database service for RAG operations.

This module provides a service for interacting with the database for RAG operations.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import asyncio
from sqlmodel import Session, select

from api_app.config import settings
from model_app.db.db import engine, Document, User, Scope

logger = logging.getLogger(__name__)

class DBService:
    """
    Service for database operations.
    
    This class provides methods to interact with the database for RAG operations.
    """
    
    def __init__(self):
        """Initialize the database service."""
        self.engine = engine
        logger.info(f"Initialized DB service with host: {settings.DB_HOST}")
            
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
            # Execute in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_rag_query, query_embedding, scope, user, limit, similarity_threshold
            )
            
            return result
        except Exception as e:
            logger.error(f"Error getting RAG documents: {str(e)}")
            return []
            
    def _execute_rag_query(
        self, 
        query_embedding: List[float],
        scope: Optional[str],
        user: Optional[str],
        limit: int,
        similarity_threshold: float
    ) -> List[Tuple[str, float]]:
        """Execute a RAG query synchronously using SQLModel."""
        with Session(self.engine) as session:
            # Start with base query
            query = select(Document)
            
            # Add filters for scope if provided
            if scope:
                # Get scope ID
                scope_stmt = select(Scope).where(Scope.name == scope)
                scope_obj = session.exec(scope_stmt).first()
                if scope_obj:
                    query = query.where(Document.scope_id == scope_obj.id)
            
            # Add filters for user if provided
            if user:
                # Get user ID
                user_stmt = select(User).where(User.username == user)
                user_obj = session.exec(user_stmt).first()
                if user_obj:
                    query = query.where(Document.user_id == user_obj.id)
            
            # Execute query to get documents
            documents = session.exec(query).all()
            
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
            return results_with_scores[:limit]
                
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            document_id: The document ID
            
        Returns:
            The document as a dictionary, or None if not found
        """
        try:
            # Execute in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_get_document, document_id
            )
            
            return result
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            return None
            
    def _execute_get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Execute a get document query synchronously using SQLModel."""
        with Session(self.engine) as session:
            try:
                # Convert string ID to int if needed
                doc_id = int(document_id) if document_id.isdigit() else document_id
                
                # Query the document
                statement = select(Document).where(Document.id == doc_id)
                document = session.exec(statement).first()
                
                if document:
                    # Convert to dictionary
                    return {
                        "id": document.id,
                        "content": document.content,
                        "embedding": document.embedding,
                        "user_id": document.user_id,
                        "scope_id": document.scope_id
                    }
                return None
            except Exception as e:
                logger.error(f"Error in _execute_get_document: {str(e)}")
                return None

# Create a global database service instance
db_service = DBService()
