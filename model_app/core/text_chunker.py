"""Module for chunking text into semantic pieces."""

from typing import List


def chunk_text(text: str) -> List[str]:
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
        # Aim for chunks of roughly 200-500 characters
        if len(current_chunk) + len(sentence) < 500:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
