"""
Dependency injection functions for FastAPI.

This module provides dependency functions that can be used in FastAPI route handlers
to inject services and other dependencies.
"""

from fastapi import HTTPException, status, Header
from typing import Optional

from api_app.config import settings
from api_app.services.chat_service import ChatService
from api_app.services.db_service import DBService
from api_app.services.ai_provider import AIProvider

def get_api_key(
    api_key: str = Header(None, alias=settings.API_KEY_NAME)
) -> Optional[str]:
    """
    Validate the API key if one is configured.
    
    Args:
        api_key: The API key from the request header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If the API key is invalid or missing when required
    """
    if settings.API_KEY is None:
        # No API key required
        return None
        
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required",
            headers={"WWW-Authenticate": settings.API_KEY_NAME},
        )
        
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": settings.API_KEY_NAME},
        )
        
    return api_key

def get_db_service() -> DBService:
    """
    Get a database service instance.
    
    Returns:
        A DBService instance
    """
    return DBService()

def get_ai_provider() -> AIProvider:
    """
    Get an AI provider instance.
    
    Returns:
        An AIProvider instance
    """
    return AIProvider()

def get_chat_service() -> ChatService:
    """
    Get a chat service instance.
    
    Returns:
        A ChatService instance
    """
    return ChatService()
