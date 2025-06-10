"""
Test script for the chat API.

This script tests the chat API endpoints.
"""

import asyncio
import httpx
import json
import uuid

# API configuration
API_URL = "http://127.0.0.1:8080/api/v1"
API_KEY = "sssss"


async def test_chat():
    """Test the chat endpoint."""
    print("\n=== Testing Chat Endpoint ===")

    # Create a session ID
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    # Create a chat request
    request_data = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": "Hello, how are you today?"}],
    }

    # Send the request
    async with httpx.AsyncClient() as client:
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["X-API-Key"] = API_KEY

        print("Sending request...")
        response = await client.post(
            f"{API_URL}/chat", json=request_data, headers=headers, timeout=30.0
        )

        # Check the response
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")


async def test_stream_chat():
    """Test the streaming chat endpoint."""
    print("\n=== Testing Streaming Chat Endpoint ===")

    # Create a session ID
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    # Create a chat request
    request_data = {
        "session_id": session_id,
        "messages": [
            {"role": "user", "content": "Tell me a short story about a robot."}
        ],
    }

    # Send the request
    async with httpx.AsyncClient() as client:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if API_KEY:
            headers["X-API-Key"] = API_KEY

        print("Sending streaming request...")
        async with client.stream(
            "POST",
            f"{API_URL}/chat/stream",
            json=request_data,
            headers=headers,
            timeout=60.0,
        ) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return

            print("Streaming response:")
            full_response = ""

            async for line in response.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue

                if line.startswith("data: "):
                    line = line[6:]  # Remove "data: " prefix

                try:
                    chunk = json.loads(line)
                    delta = chunk.get("delta", "")
                    print(delta, end="", flush=True)
                    full_response += delta
                except Exception as e:
                    print(f"\nError parsing chunk: {str(e)}")

            print("\n\nFull response:", full_response)


async def test_sessions():
    """Test the sessions endpoints."""
    print("\n=== Testing Sessions Endpoints ===")

    # Create a session ID
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    # Create a chat request to populate the session
    request_data = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": "This is a test message."}],
    }

    async with httpx.AsyncClient() as client:
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["X-API-Key"] = API_KEY

        # Send a chat request to create the session
        print("Creating session...")
        response = await client.post(
            f"{API_URL}/chat", json=request_data, headers=headers
        )

        if response.status_code != 200:
            print(f"Error creating session: {response.status_code}")
            return

        # Get session info
        print("\nGetting session info...")
        response = await client.get(
            f"{API_URL}/chat/sessions/{session_id}", headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Session info: {json.dumps(result, indent=2)}")
        else:
            print(f"Error getting session: {response.status_code}")
            print(f"Response: {response.text}")

        # List all sessions
        print("\nListing all sessions...")
        response = await client.get(f"{API_URL}/chat/sessions", headers=headers)

        if response.status_code == 200:
            result = response.json()
            print(f"Found {len(result)} sessions")
            print(f"Sessions: {json.dumps(result, indent=2)}")
        else:
            print(f"Error listing sessions: {response.status_code}")
            print(f"Response: {response.text}")

        # Clear the session
        print("\nClearing session...")
        response = await client.delete(
            f"{API_URL}/chat/sessions/{session_id}", headers=headers
        )

        if response.status_code == 204:
            print("Session cleared successfully")
        else:
            print(f"Error clearing session: {response.status_code}")
            print(f"Response: {response.text}")


async def run_tests():
    """Run all tests."""
    await test_chat()
    await test_stream_chat()
    await test_sessions()


if __name__ == "__main__":
    """Run the tests."""
    asyncio.run(run_tests())
