#!/bin/bash

# Run the application with Docker Compose

# Exit on error
set -e

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found."
    echo "Please create a .env file based on .env-example."
    exit 1
fi

# Run with Docker Compose
echo "Starting services with Docker Compose..."
docker-compose up --build

# Note: Use Ctrl+C to stop the services
