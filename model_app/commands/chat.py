import os
import logging
from typing import Optional, List, Tuple
from openai import AsyncOpenAI
from model_app.core.rag import rag_query

# Configure logging
logger = logging.getLogger(__name__)

# Configure OpenAI client for chat model
chat_client = AsyncOpenAI(
    api_key="dummy-key",
    base_url=os.getenv("CHAT_MODEL_URL", "http://localhost:8005/v1"),
)


def build_rag_prompt(chunks: List[str], user_query: str) -> str:
    """Build a prompt for the chat model using RAG context chunks.

    Args:
        chunks: List of context strings
        user_query: The user's question

    Returns:
        Formatted prompt string
    """
    labeled_chunks = "\n\n".join(
        [f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(chunks)]
    )
    return f"""You are a helpful assistant answering user questions using the information provided below. 
        Use only the context to answer. If the answer is not contained in the provided context, 
        reply with "The information is not available in the provided context."

        --- CONTEXT CHUNKS ---
        {labeled_chunks}
        --- END OF CONTEXT ---

        --- USER QUESTION ---
        {user_query}
        --- END OF QUESTION ---

        Provide a clear and concise answer based only on the context above in the language of user question.
        """


async def get_chat_response(
    question: str, username: Optional[str] = None, scope_name: Optional[str] = None
) -> Tuple[str, List[str]]:
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
    )

    full_docs = [doc[0] for doc in documents_found]
    logger.info(f"Using {len(full_docs)} documents for response")

    response = await chat_client.completions.create(
        model="chat-model",
        prompt=build_rag_prompt(full_docs, question),
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].text, full_docs


async def chat(username: str = "", scope_name: str = ""):
    """Interactive chat interface (maintains backward compatibility with app.py)."""
    print("Chat started. Type 'exit' to end the chat.")
    print(f"Optional filters - scope: {scope_name or 'all'}, user: {username or 'any'}")

    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break

        answer, _ = await get_chat_response(
            question, username or None, scope_name or None
        )
        print(f"You Asked: {question}")
        print(f"Answer: {answer}")

    print("Chat ended.")
