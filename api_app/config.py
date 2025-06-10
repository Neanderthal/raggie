"""
Configuration settings for the chat API.

This module defines the configuration settings for the chat API.
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings.

    This class defines the configuration settings for the application.
    Settings can be overridden by environment variables.
    """

    # API settings
    API_TITLE: str = "Chat API"
    API_DESCRIPTION: str = "API for chat interactions with AI models"
    API_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"

    # Security settings
    API_KEY: Optional[str] = None
    API_KEY_NAME: str = "X-API-Key"

    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["*"])
    CORS_METHODS: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    CORS_HEADERS: List[str] = Field(default=["*"])

    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "pgvector_rag"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"

    # AI model settings
    CHAT_MODEL_URL: str = "http://localhost:8000"
    CHAT_MODEL_NAME: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL_URL: str = "http://localhost:8000"
    EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"

    RATE_LIMIT_PER_MINUTE: int = 40
    SESSION_EXPIRY_MINUTES: int = 320
    # Logging settings
    LOG_LEVEL: str = "INFO"

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = True


# Create a global settings instance
settings = Settings()
