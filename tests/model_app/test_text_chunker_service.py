"""Tests for text chunker service."""

import pytest
from model_app.core.text_chunker_service import TextChunker, chunk_text_legacy
from model_app.core.rag_config import RAGConfig


class TestTextChunkerService:
    """Test suite for text chunker service."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = RAGConfig()
        config.chunk_size = 100
        config.chunk_overlap = 20
        return config
    
    @pytest.fixture
    def chunker(self, config):
        """Create a text chunker with test configuration."""
        return TextChunker(config)
    
    def test_chunk_text_small(self, chunker):
        """Test chunking of small text."""
        text = "This is a short sentence."
        chunks = chunker.chunk_text(text)
        assert chunks == ["This is a short sentence"]
    
    def test_chunk_text_large(self, chunker):
        """Test chunking of large text."""
        # Create text larger than chunk size
        text = ". ".join(["This is sentence"] * 10)
        chunks = chunker.chunk_text(text)
        
        # Verify multiple chunks were created
        assert len(chunks) > 1
        
        # Verify chunks are smaller than configured size
        assert all(len(chunk) <= chunker.config.chunk_size for chunk in chunks)
    
    def test_chunk_text_empty(self, chunker):
        """Test chunking of empty text."""
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
    
    def test_chunk_text_with_overlap(self, chunker):
        """Test chunking with overlap."""
        # Create text large enough for multiple chunks
        text = " ".join(["word"] * 50)
        chunks = chunker.chunk_text_with_overlap(text)
        
        # Verify multiple chunks were created
        assert len(chunks) > 1
        
        # With overlap, we should have more chunks than with regular chunking
        regular_chunks = chunker.chunk_text(text)
        assert len(chunks) >= len(regular_chunks)


def test_chunk_text_legacy():
    """Test the legacy chunk_text function."""
    text = "This is a test sentence. This is another test sentence."
    chunks = chunk_text_legacy(text)
    assert len(chunks) > 0
    assert "This is a test sentence" in chunks[0]
