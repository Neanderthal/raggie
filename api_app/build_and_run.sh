#!/bin/bash

# Build and run the Docker container for the chat API

# Exit on error
set -e

# Build the Docker image
echo "Building Docker image..."
docker build -t chat-api .

# Run the container
echo "Running container..."
docker run -p 8080:8080 \
  --name chat-api \
  --env-file .env \
  --rm \
  chat-api

echo "Container started. API is available at http://localhost:8000"
echo "Press Ctrl+C to stop the container."
