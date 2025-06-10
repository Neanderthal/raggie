# Chat API with RAG

A FastAPI-based chat API with Retrieval-Augmented Generation (RAG) capabilities, designed to run in a Docker container.

## Features

- **RESTful API**: Clean, well-documented API endpoints for chat interactions
- **Streaming Support**: Real-time streaming of AI responses
- **Session Management**: Maintain conversation history and context
- **RAG Integration**: Enhance responses with relevant documents from a vector database
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Security**: API key authentication and rate limiting
- **Scalability**: Designed for horizontal scaling

## Architecture

The application follows a clean architecture pattern with the following components:

- **API Layer**: FastAPI routes and endpoints
- **Service Layer**: Business logic and integration
- **Data Layer**: Database interactions and persistence
- **Schemas**: Pydantic models for validation and serialization
- **Utilities**: Helper functions and utilities

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Environment Setup

1. Copy the example environment file:

```bash
cp .env-example .env
```

2. Edit the `.env` file to configure your settings.

### Running with Docker Compose

The easiest way to run the application is with Docker Compose:

```bash
./run_with_compose.sh
```

This will start:
- The API service
- A PostgreSQL database with pgvector extension
- An AI model service (placeholder)

### Running the API Only

To run just the API service:

```bash
./build_and_run.sh
```

### Local Development

For local development:

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the development server:

```bash
python run_dev.py
```

## API Endpoints

### Chat Endpoints

- `POST /api/v1/chat`: Send a message and get a response
- `POST /api/v1/chat/stream`: Send a message and get a streaming response

### Session Management

- `GET /api/v1/chat/sessions`: List all active chat sessions
- `GET /api/v1/chat/sessions/{session_id}`: Get information about a specific session
- `DELETE /api/v1/chat/sessions/{session_id}`: Clear a session's history

## Testing

Run the test script to verify the API is working correctly:

```bash
python test_api.py
```

## Configuration

The application can be configured through environment variables or the `.env` file. See `.env-example` for available options.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
