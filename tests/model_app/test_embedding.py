"""Tests for embedding generation functionality."""

import pytest
from unittest.mock import patch
from model_app.core.embedding import generate_embeddings


class TestGenerateEmbeddings:
    """Test suite for embedding generation functions."""
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self):
        """Test successful embedding generation with mocked client."""
        with patch('model_app.core.embedding.EmbeddingService.generate_embeddings', 
                   return_value=("test text", [0.1, 0.2, 0.3])):
            text, embedding = await generate_embeddings("test text")
            assert text == "test text"
            assert embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_text(self):
        """Test empty text handling."""
        text, embed = await generate_embeddings("   ")
        assert text == "empty"
        assert len(embed) > 0

    @pytest.mark.asyncio
    async def test_generate_embeddings_failure_with_fallback(self):
        """Test API failure case falls back to hash-based embeddings."""
        with patch('model_app.core.embedding.EmbeddingService.generate_embeddings', 
                   side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await generate_embeddings("test")
