import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Dict, Any
import pytest_asyncio

from model_app.tasks.text_to_embedings_task import (
    EmbeddingTaskProcessor,
    texts_to_embeddings,
    EmbeddingTaskConfig
)
from model_app.core.embedding import EmbeddingService, EmbeddingConnectionError, EmbeddingAPIError


class TestEmbeddingTaskProcessor:
    """Integration tests for the EmbeddingTaskProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a processor with mocked dependencies."""
        with patch('model_app.tasks.text_to_embedings_task.EmbeddingService') as mock_service_cls:
            # Setup the mock embedding service
            mock_service = MagicMock()
            mock_service_cls.return_value = mock_service
            
            # Create the processor
            processor = EmbeddingTaskProcessor()
            
            # Replace the real embedding service with our mock
            processor.embedding_service = mock_service
            
            yield processor

    @pytest.fixture
    def mock_db_functions(self):
        """Mock database functions."""
        with patch('model_app.tasks.text_to_embedings_task.get_or_create_user') as mock_get_user, \
             patch('model_app.tasks.text_to_embedings_task.get_or_create_scope') as mock_get_scope, \
             patch('model_app.tasks.text_to_embedings_task.create_initial_document') as mock_create_doc, \
             patch('model_app.tasks.text_to_embedings_task.store_embeddings') as mock_store:
            
            # Configure mocks
            mock_get_user.return_value = 1  # User ID
            mock_get_scope.return_value = 2  # Scope ID
            mock_create_doc.return_value = 3  # Document ID
            
            # Use AsyncMock to return a list directly instead of a Future
            mock_store.return_value = ["doc1", "doc2"]
            
            yield {
                "get_user": mock_get_user,
                "get_scope": mock_get_scope,
                "create_doc": mock_create_doc,
                "store": mock_store
            }

    @pytest_asyncio.fixture
    async def event_loop(self):
        """Create an event loop for tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.mark.asyncio
    async def test_process_texts_to_embeddings_success(self, processor, mock_db_functions):
        """Test successful processing of texts to embeddings."""
        # Setup
        texts = ["This is text 1", "This is text 2"]
        username = "testuser"
        scope_name = "testscope"
        document_name = "testdoc"
        
        # Mock embedding generation
        processor.embedding_service.generate_embeddings = AsyncMock()
        processor.embedding_service.generate_embeddings.side_effect = [
            ("this is text 1", [0.1, 0.2, 0.3]),
            ("this is text 2", [0.4, 0.5, 0.6])
        ]
        
        # Execute
        await processor.process_texts_to_embeddings(
            texts, username, scope_name, document_name
        )
        
        # Verify
        assert processor.embedding_service.generate_embeddings.call_count == 2
        mock_db_functions["get_user"].assert_called_once_with(username)
        mock_db_functions["get_scope"].assert_called_once_with(scope_name)
        mock_db_functions["create_doc"].assert_called_once()
        mock_db_functions["store"].assert_called_once()
        
        # Verify the embeddings data passed to store_embeddings
        call_args = mock_db_functions["store"].call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["text"] == "This is text 1"
        assert call_args[0]["embedding"] == [0.1, 0.2, 0.3]
        assert call_args[0]["metadata"]["username"] == username
        assert call_args[0]["metadata"]["scope"] == scope_name
        assert call_args[0]["metadata"]["document_name"] == document_name

    @pytest.mark.asyncio
    async def test_process_texts_to_embeddings_with_document_id(self, processor, mock_db_functions):
        """Test processing with document ID."""
        # Setup
        texts = ["This is text 1"]
        username = "testuser"
        scope_name = "testscope"
        document_name = "testdoc"
        document_id = "doc123"
        
        # Mock embedding generation
        processor.embedding_service.generate_embeddings = AsyncMock()
        processor.embedding_service.generate_embeddings.return_value = ("this is text 1", [0.1, 0.2, 0.3])
        
        # Execute
        await processor.process_texts_to_embeddings(
            texts, username, scope_name, document_name, document_id
        )
        
        # Verify document ID in metadata
        call_args = mock_db_functions["store"].call_args[0][0]
        assert call_args[0]["metadata"]["document_id"] == document_id

    @pytest.mark.asyncio
    async def test_process_texts_to_embeddings_embedding_error(self, processor, mock_db_functions):
        """Test handling of embedding service errors."""
        # Setup
        texts = ["This is text 1", "This is text 2"]
        username = "testuser"
        scope_name = "testscope"
        
        # Mock embedding generation with error
        processor.embedding_service.generate_embeddings = AsyncMock()
        processor.embedding_service.generate_embeddings.side_effect = [
            EmbeddingAPIError("API error"),
            ("this is text 2", [0.4, 0.5, 0.6])
        ]
        
        # Execute
        await processor.process_texts_to_embeddings(
            texts, username, scope_name
        )
        
        # Verify only one embedding was stored
        call_args = mock_db_functions["store"].call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["text"] == "This is text 2"

    @pytest.mark.asyncio
    async def test_process_texts_to_embeddings_connection_error(self, processor, mock_db_functions):
        """Test handling of connection errors."""
        # Setup
        texts = ["This is text 1"]
        username = "testuser"
        scope_name = "testscope"
        
        # Mock embedding generation with connection error
        processor.embedding_service.generate_embeddings = AsyncMock()
        processor.embedding_service.generate_embeddings.side_effect = EmbeddingConnectionError("Connection error")
        
        # Execute and verify exception is raised
        with pytest.raises(EmbeddingConnectionError):
            await processor.process_texts_to_embeddings(
                texts, username, scope_name
            )
        
        # Verify no embeddings were stored
        mock_db_functions["store"].assert_not_called()

    @pytest.mark.asyncio
    async def test_process_texts_to_embeddings_empty_result(self, processor, mock_db_functions):
        """Test handling of empty embedding results."""
        # Setup
        texts = ["This is text 1"]
        username = "testuser"
        scope_name = "testscope"
        
        # Mock embedding generation with all errors
        processor.embedding_service.generate_embeddings = AsyncMock()
        processor.embedding_service.generate_embeddings.side_effect = EmbeddingAPIError("API error")
        
        # Execute
        await processor.process_texts_to_embeddings(
            texts, username, scope_name
        )
        
        # Verify no embeddings were stored
        mock_db_functions["store"].assert_not_called()


@patch('model_app.tasks.text_to_embedings_task.asyncio.run')
@patch('model_app.tasks.text_to_embedings_task.embedding_processor')
def test_texts_to_embeddings_task(mock_processor, mock_run):
    """Test the Celery task wrapper."""
    # Setup
    texts = ["Test text"]
    username = "testuser"
    scope_name = "testscope"
    document_name = "testdoc"
    document_id = "doc123"
    
    # Execute
    texts_to_embeddings(
        self=None,  # Celery task is bound, but we're calling directly
        texts=texts,
        username=username,
        scope_name=scope_name,
        document_name=document_name,
        document_id=document_id
    )
    
    # Verify
    mock_run.assert_called_once()
    process_call = mock_run.call_args[0][0]
    assert asyncio.iscoroutine(process_call)  # Verify it's a coroutine
