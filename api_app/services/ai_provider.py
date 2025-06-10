"""
AI provider service for chat interactions.

This module provides a service for interacting with AI models for chat and embeddings.
"""

import logging
from typing import List, Dict, AsyncGenerator, Optional
from model_app.commands.chat import get_chat_response
from model_app.core.embedding import generate_embeddings

logger = logging.getLogger(__name__)


class AIProvider:
    """
    Service for interacting with AI models.

    This class provides methods to generate chat completions and embeddings
    using configured AI models.
    """

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for text using the core RAG implementation.

        Args:
            text: The text to get embeddings for

        Returns:
            List of embedding values

        Raises:
            Exception: If embedding generation fails
        """
        try:
            # Get embeddings from containerized service
            _, embedding = await generate_embeddings(text)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    async def get_chat_response(
        self,
        messages: List[Dict[str, str]],
        context_docs: Optional[List[str]] = None,
        username: Optional[str] = None,
        scope: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Get a chat response with RAG context.

        Args:
            messages: The conversation messages
            context_docs: Optional list of context documents for RAG
            username: Optional username for filtering documents
            scope: Optional scope for filtering documents
            temperature: The sampling temperature (not currently used)
            max_tokens: The maximum number of tokens to generate (not currently used)

        Returns:
            The generated text response

        Raises:
            Exception: If the chat completion request fails
        """
        try:
            # Extract the last user message as the question
            last_message = next(
                (
                    msg["content"]
                    for msg in reversed(messages)
                    if msg.get("role", "") == "user"
                ),
                None,
            )
            if not last_message:
                raise ValueError("No user message found in conversation")

            # Get response using core RAG implementation
            response, _ = await get_chat_response(
                question=last_message, username=username, scope_name=scope
            )
            return response

        except Exception as e:
            logger.error(f"Error getting chat response: {str(e)}")
            raise

    async def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        context_docs: Optional[List[str]] = None,
        username: Optional[str] = None,
        scope: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response with RAG context.

        Note: Currently returns the full response at once since the core
        implementation doesn't support streaming yet.

        Args:
            messages: The conversation messages
            username: Optional username for filtering documents
            scope: Optional scope for filtering documents
            temperature: The sampling temperature (not currently used)
            max_tokens: The maximum number of tokens to generate (not currently used)

        Yields:
            The complete generated text as a single chunk

        Raises:
            Exception: If the chat completion request fails
        """
        try:
            response = await self.get_chat_response(
                messages=messages,
                username=username,
                scope=scope,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            yield response

        except Exception as e:
            logger.error(f"Error streaming chat response: {str(e)}")
            raise
