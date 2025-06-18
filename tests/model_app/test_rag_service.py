"""Tests for RAG service functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Tuple, Dict, Any
import pytest_asyncio
import asyncio

from model_app.core.rag_service import RAGService, rag_query, store_embeddings
from model_app.core.rag_exceptions import RAGQueryError, RAGStorageError
from langchain_core.documents import Document


class TestRAGService:
    """Test suite for RAG service."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = MagicMock()
        return mock_store
    
    @pytest.fixture
    def rag_service(self, mock_vector_store):
        """Create a RAG service with mocked vector store."""
        service = RAGService()
        service._vector_store = mock_vector_store
        return service
    
    @pytest_asyncio.fixture
    async def event_loop(self):
        """Create an event loop for tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
    @pytest.mark.asyncio
    async def test_query_documents_success(self, rag_service, mock_vector_store):
        """Test successful document query."""
        # Setup mock response
        doc1 = Document(page_content="Test document 1", metadata={"scope": "test"})
        doc2 = Document(page_content="Test document 2", metadata={"scope": "test"})
        mock_vector_store.similarity_search_with_score.return_value = [
            (doc1, 0.9),
            (doc2, 0.8)
        ]
        
        # Execute query
        results = await rag_service.query_documents(
            query="test query",
            scope="test",
            k=2,
            similarity_threshold=0.7
        )
        
        # Verify results
        assert len(results) == 2
        assert results[0][0] == "Test document 1"
        assert results[0][1] == 0.9
        assert results[1][0] == "Test document 2"
        assert results[1][1] == 0.8
        
        # Verify filter was applied
        mock_vector_store.similarity_search_with_score.assert_called_once()
        call_args = mock_vector_store.similarity_search_with_score.call_args
        assert call_args[1]["filter"]["scope"]["$eq"] == "test"
        assert call_args[1]["k"] == 2
    
    @pytest.mark.asyncio
    async def test_query_documents_with_threshold_filtering(self, rag_service, mock_vector_store):
        """Test query with threshold filtering."""
        # Setup mock response with scores above and below threshold
        doc1 = Document(page_content="Test document 1", metadata={})
        doc2 = Document(page_content="Test document 2", metadata={})
        doc3 = Document(page_content="Test document 3", metadata={})
        mock_vector_store.similarity_search_with_score.return_value = [
            (doc1, 0.9),
            (doc2, 0.6),
            (doc3, 0.3)
        ]
        
        # Execute query with threshold 0.7
        results = await rag_service.query_documents(
            query="test query",
            similarity_threshold=0.7
        )
        
        # Verify only results above threshold are returned
        assert len(results) == 1
        assert results[0][0] == "Test document 1"
        assert results[0][1] == 0.9
    
    @pytest.mark.asyncio
    async def test_query_documents_error_handling(self, rag_service, mock_vector_store):
        """Test error handling during query."""
        # Setup mock to raise exception
        mock_vector_store.similarity_search_with_score.side_effect = ValueError("Invalid query")
        
        # Execute and verify exception is raised
        with pytest.raises(RAGQueryError):
            await rag_service.query_documents(query="test query")
    
    @pytest.mark.asyncio
    async def test_store_embeddings_success(self, rag_service, mock_vector_store):
        """Test successful embedding storage."""
        # Setup
        embeddings_data = [
            {
                "text": "Test text 1",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"scope": "test", "username": "testuser"}
            },
            {
                "text": "Test text 2",
                "embedding": [0.4, 0.5, 0.6],
                "metadata": {"scope": "test", "username": "testuser"}
            }
        ]
        
        # Mock vector store response
        mock_vector_store.add_documents.return_value = ["id1", "id2"]
        
        # Mock db function
        with patch('model_app.core.rag_service.store_vector_document_links') as mock_store_links:
            # Execute
            result_ids = await rag_service.store_embeddings(
                embeddings_data=embeddings_data,
                initial_document_id=123
            )
            
            # Verify
            assert result_ids == ["id1", "id2"]
            assert mock_vector_store.add_documents.call_count == 1
            mock_store_links.assert_called_once_with(123, ["id1", "id2"])
    
    @pytest.mark.asyncio
    async def test_store_embeddings_error_handling(self, rag_service, mock_vector_store):
        """Test error handling during embedding storage."""
        # Setup
        embeddings_data = [{"text": "Test", "embedding": [0.1], "metadata": {}}]
        
        # Mock vector store to raise exception
        mock_vector_store.add_documents.side_effect = Exception("Storage error")
        
        # Execute and verify exception is raised
        with pytest.raises(RAGStorageError):
            await rag_service.store_embeddings(embeddings_data)


@pytest.mark.asyncio
@patch('model_app.core.rag_service.rag_service.query_documents')
async def test_rag_query_convenience_function(mock_query):
    """Test the rag_query convenience function."""
    # Setup
    mock_query.return_value = [("Test document", 0.9)]
    
    # Execute
    result = await rag_query("test", scope="test", k=5)
    
    # Verify
    assert result == [("Test document", 0.9)]
    mock_query.assert_called_once_with("test", "test", None, None, 5, 0.2)


@pytest.mark.asyncio
@patch('model_app.core.rag_service.rag_service.store_embeddings')
async def test_store_embeddings_convenience_function(mock_store):
    """Test the store_embeddings convenience function."""
    # Setup
    mock_store.return_value = ["id1"]
    embeddings_data = [{"text": "Test", "embedding": [0.1], "metadata": {}}]
    
    # Execute
    result = await store_embeddings(embeddings_data, 123)
    
    # Verify
    assert result == ["id1"]
    mock_store.assert_called_once_with(embeddings_data, 123)
