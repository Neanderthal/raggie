import pytest

# Configure pytest-asyncio to use the "strict" mode
pytest_plugins = ["pytest_asyncio"]

# Set default event loop policy for all async tests
def pytest_configure(config):
    """Configure pytest."""
    import asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
