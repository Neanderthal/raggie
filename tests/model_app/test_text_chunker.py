"""Tests for text chunker module."""

import pytest
from model_app.core.text_chunker import chunk_text


class TestChunkText:
    """Test suite for text chunking functions."""
    
    def test_chunk_text_small(self):
        """Test small text chunking."""
        text = "This is a short sentence."
        chunks = chunk_text(text)
        assert chunks == ["This is a short sentence"]

    def test_chunk_text_large(self):
        """Test large text chunking."""
        text = ". ".join(["This is sentence"] * 50)
        chunks = chunk_text(text)
        assert len(chunks) > 1
        assert all(len(chunk) <= 500 for chunk in chunks)

    def test_chunk_text_empty(self):
        """Test empty text chunking."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []
