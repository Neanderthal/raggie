"""Service for advanced text chunking operations."""

import logging
from typing import List, Optional
from model_app.core.rag_config import rag_config, RAGConfig
from model_app.core.text_chunker import chunk_text as basic_chunk_text

logger = logging.getLogger(__name__)


class TextChunker:
    """Service for chunking text into semantic pieces."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or rag_config

    def chunk_text(self, text: str) -> List[str]:
        """Split text into semantic chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        chunks = []
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        current_chunk = ""
        for sentence in sentences:
            # Use configurable chunk size
            if len(current_chunk) + len(sentence) < self.config.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_text_with_overlap(self, text: str) -> List[str]:
        """Split text into chunks with overlap for better context preservation.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks with overlap
        """
        if not text.strip():
            return []

        # Simple implementation - can be enhanced with more sophisticated chunking
        words = text.split()
        chunks = []
        
        words_per_chunk = self.config.chunk_size // 5  # Rough estimate
        overlap_words = self.config.chunk_overlap // 5
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
                
        return chunks


# Global chunker instance
text_chunker = TextChunker()


# Convenience function for backward compatibility
def chunk_text_legacy(text: str) -> List[str]:
    """Split text into semantic chunks using the default chunker."""
    return basic_chunk_text(text)
