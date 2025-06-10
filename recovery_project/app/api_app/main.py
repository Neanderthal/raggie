"""
Main application module.

This module sets up the FastAPI application.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api_app.config import settings
from api_app.api.endpoints import chat

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        The configured FastAPI application
    """
    # Create the FastAPI app
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )
    
    # Include routers
    app.include_router(chat.router, prefix=settings.API_PREFIX)
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Run startup tasks."""
        logger.info("Starting up the application")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Run shutdown tasks."""
        logger.info("Shutting down the application")
    
    return app

app = create_app()

if __name__ == "__main__":
    """Run the application with uvicorn when executed directly."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
