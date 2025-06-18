import pytest
import asyncio
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Dict, Any
import pytest_asyncio

from model_app.core.embedding import (
    EmbeddingService,
    EmbeddingConfig,
    CustomLlamaEmbeddings,
    EmbeddingConnectionError,
    EmbeddingAPIError,
    clean_text
)


class TestEmbeddingService:
    """Integration tests for the EmbeddingService."""
    
    @pytest_asyncio.fixture
    async def event_loop(self):
        """Create an event loop for tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = EmbeddingConfig()
        config.base_url = "http://test-embedding-service:8000/v1"
        config.max_retries = 2
        config.base_retry_delay = 0.1
        config.health_check_timeout = 0.5
        return config

    @pytest.fixture
    def service(self, config):
        """Create an embedding service with mocked embedding model."""
        with patch('model_app.core.embedding.CustomLlamaEmbeddings') as mock_embeddings_cls:
            # Setup mock embedding model
            mock_model = MagicMock()
            mock_embeddings_cls.return_value = mock_model
            
            # Create service
            service = EmbeddingService(config)
            
            # Replace the real model with our mock
            service.embedding_model = mock_model
            
            yield service

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, service):
        """Test successful embedding generation."""
        # Setup
        text = "This is a test text"
        expected_embedding = [0.1, 0.2, 0.3]
        
        # Mock the embedding model
        service.embedding_model.aembed_documents = AsyncMock()
        service.embedding_model.aembed_documents.return_value = [expected_embedding]
        
        # Mock health check
        service._check_service_health = AsyncMock()
        
        # Execute
        clean_text, embedding = await service.generate_embeddings(text)
        
        # Verify
        assert clean_text == "this is a test text"
        assert embedding == expected_embedding
        service.embedding_model.aembed_documents.assert_called_once()
        service._check_service_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_text(self, service):
        """Test handling of empty text."""
        # Setup
        text = "   "  # Just whitespace
        
        # Execute
        clean_text, embedding = await service.generate_embeddings(text)
        
        # Verify
        assert clean_text == "empty"
        assert embedding == [0.0] * service.config.default_embedding_dimension
        service.embedding_model.aembed_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_connection_error_with_retry(self, service):
        """Test retry logic on connection error."""
        # Setup
        text = "This is a test text"
        expected_embedding = [0.1, 0.2, 0.3]
        
        # Mock health check
        service._check_service_health = AsyncMock()
        
        # Mock embedding model to fail once then succeed
        service.embedding_model.aembed_documents = AsyncMock()
        service.embedding_model.aembed_documents.side_effect = [
            EmbeddingConnectionError("Connection failed"),
            [expected_embedding]
        ]
        
        # Mock backoff to avoid waiting
        service._wait_with_backoff = AsyncMock()
        
        # Execute
        clean_text, embedding = await service.generate_embeddings(text)
        
        # Verify
        assert clean_text == "this is a test text"
        assert embedding == expected_embedding
        assert service.embedding_model.aembed_documents.call_count == 2
        service._wait_with_backoff.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_connection_error_max_retries(self, service):
        """Test max retries exceeded with connection error."""
        # Setup
        text = "This is a test text"
        
        # Mock health check
        service._check_service_health = AsyncMock()
        
        # Mock embedding model to always fail
        service.embedding_model.aembed_documents = AsyncMock()
        service.embedding_model.aembed_documents.side_effect = EmbeddingConnectionError("Connection failed")
        
        # Mock backoff to avoid waiting
        service._wait_with_backoff = AsyncMock()
        
        # Execute and verify exception is raised
        with pytest.raises(EmbeddingConnectionError):
            await service.generate_embeddings(text)
        
        # Verify retry attempts
        assert service.embedding_model.aembed_documents.call_count == service.config.max_retries
        assert service._wait_with_backoff.call_count == service.config.max_retries - 1

    @pytest.mark.asyncio
    async def test_generate_embeddings_api_error_with_retry(self, service):
        """Test retry logic on API error."""
        # Setup
        text = "This is a test text"
        expected_embedding = [0.1, 0.2, 0.3]
        
        # Mock health check
        service._check_service_health = AsyncMock()
        
        # Mock embedding model to fail once then succeed
        service.embedding_model.aembed_documents = AsyncMock()
        service.embedding_model.aembed_documents.side_effect = [
            EmbeddingAPIError("API error"),
            [expected_embedding]
        ]
        
        # Mock backoff to avoid waiting
        service._wait_with_backoff = AsyncMock()
        
        # Execute
        clean_text, embedding = await service.generate_embeddings(text)
        
        # Verify
        assert clean_text == "this is a test text"
        assert embedding == expected_embedding
        assert service.embedding_model.aembed_documents.call_count == 2
        service._wait_with_backoff.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_api_error_max_retries(self, service):
        """Test max retries exceeded with API error."""
        # Setup
        text = "This is a test text"
        
        # Mock health check
        service._check_service_health = AsyncMock()
        
        # Mock embedding model to always fail
        service.embedding_model.aembed_documents = AsyncMock()
        service.embedding_model.aembed_documents.side_effect = EmbeddingAPIError("API error")
        
        # Mock backoff to avoid waiting
        service._wait_with_backoff = AsyncMock()
        
        # Execute
        clean_text, embedding = await service.generate_embeddings(text)
        
        # Verify fallback embedding is returned
        assert clean_text == "this is a test text"
        assert embedding == [0.0] * service.config.default_embedding_dimension
        assert service.embedding_model.aembed_documents.call_count == service.config.max_retries
        assert service._wait_with_backoff.call_count == service.config.max_retries - 1

    @pytest.mark.asyncio
    async def test_check_service_health_success(self, service):
        """Test successful health check."""
        # Setup
        with patch('model_app.core.embedding.httpx.AsyncClient') as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            
            # Execute
            await service._check_service_health()
            
            # Verify
            mock_client.get.assert_called_once()
            assert "health" in mock_client.get.call_args[0][0]

    @pytest.mark.asyncio
    async def test_check_service_health_failure(self, service):
        """Test failed health check."""
        # Setup
        with patch('model_app.core.embedding.httpx.AsyncClient') as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.get.return_value = mock_response
            
            # Execute
            await service._check_service_health()
            
            # Verify - should log warning but not raise exception
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_service_health_exception(self, service):
        """Test exception during health check."""
        # Setup
        with patch('model_app.core.embedding.httpx.AsyncClient') as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            
            # Execute
            await service._check_service_health()
            
            # Verify - should log warning but not raise exception
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_with_backoff(self, service):
        """Test exponential backoff calculation."""
        # Setup
        with patch('model_app.core.embedding.asyncio.sleep') as mock_sleep:
            # Execute
            await service._wait_with_backoff(0)
            await service._wait_with_backoff(1)
            await service._wait_with_backoff(10)  # Should hit max cap
            
            # Verify
            assert mock_sleep.call_count == 3
            assert mock_sleep.call_args_list[0][0][0] == service.config.base_retry_delay
            assert mock_sleep.call_args_list[1][0][0] == service.config.base_retry_delay * 2
            assert mock_sleep.call_args_list[2][0][0] == service.config.max_retry_delay


def test_clean_text():
    """Test text cleaning function."""
    # Test whitespace handling
    assert clean_text("  test  ") == "test"
    assert clean_text("test\t\ttext") == "test text"
    
    # Test newline handling
    assert clean_text("test\n\ntext") == "test\ntext"
    assert clean_text("test\n \n text") == "test\ntext"
    
    # Test case conversion
    assert clean_text("Test TEXT") == "test text"
    
    # Test empty input
    assert clean_text("") == ""
    assert clean_text("   ") == ""
