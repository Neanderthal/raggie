"""
Development server script.

This script runs the application in development mode.
"""

import uvicorn

if __name__ == "__main__":
    """Run the application with uvicorn in development mode."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
    )
