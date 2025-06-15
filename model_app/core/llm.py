import os
import logging
from typing import Optional, List, Tuple
from openai import AsyncOpenAI
from model_app.core.rag import rag_query
from dotenv import load_dotenv

# Load environment variables from the model_app directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Configure logging
logger = logging.getLogger(__name__)

# Configure OpenAI client for chat model
# Use different URLs for Docker vs local development
# Initialize client lazily to avoid import-time errors


chat_client = None


def get_chat_client():
    global chat_client
    if chat_client is None:
        chat_client = AsyncOpenAI(
            api_key="dummy-key", base_url=os.getenv("CHAT_MODEL_URL"), timeout=30.0
        )
    return chat_client


def build_system_prompt() -> str:
    return """You are a helpful assistant answering user questions using the provided context information.
Use only the context to answer. If the answer is not contained in the provided context,
reply with "The information is not available in the provided context."
Provide clear and concise answers in the same language as the user's question."""


def build_user_prompt(chunks: List[str], user_query: str) -> str:
    """Build a prompt for the chat model using RAG context chunks."""
    context = "\n\n".join([f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(chunks)])
    return f"""Here is the relevant context information:

{context}

Question: {user_query}"""


async def get_chat_response(
    question: str, username: Optional[str] = None, scope_name: Optional[str] = None
) -> Tuple[str, List[str]]:
    if not question:
        raise ValueError("Question cannot be empty")
    """Get a chat response using RAG (function call interface).

    Args:
        question: The user's question
        username: Optional username filter
        scope_name: Optional scope filter

    Returns:
        Tuple of (answer, list of document contents used)
    """
    documents_found = await rag_query(
        query=question,
        scope=scope_name,
        user=username,
        k=3,  # Get top 3 most relevant documents
    )
    full_docs = [doc[0] for doc in documents_found]  # Extract just the document content
    logger.info(f"Using {len(full_docs)} documents for response")

    try:
        response = await get_chat_client().chat.completions.create(
            model="chat-model",
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(full_docs, question)},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        if not response.choices or not response.choices[0].message.content:
            return "No response from the chat model", full_docs
        return response.choices[0].message.content.strip(), full_docs
    except Exception as e:
        logger.error(f"Chat request failed: {str(e)}")
        return "An error occurred while generating the response", full_docs
