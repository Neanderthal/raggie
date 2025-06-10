"""
Pydantic schemas for chat API.

This module defines the Pydantic schemas used in the chat API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import uuid4

class Message(BaseModel):
    """
    A chat message.
    
    Attributes:
        role: The role of the message sender (user, assistant, system)
        content: The content of the message
    """
    role: str
    content: str

class ChatRequest(BaseModel):
    """
    A chat request.
    
    Attributes:
        session_id: The session ID
        messages: The list of messages
        scope: Optional scope for RAG
        username: Optional username for RAG
    """
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    messages: List[Message]
    scope: Optional[str] = None
    username: Optional[str] = None

class ChatResponse(BaseModel):
    """
    A chat response.
    
    Attributes:
        message: The response message
        session_id: The session ID
        created_at: The creation time
    """
    message: Message
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)

class StreamChunk(BaseModel):
    """
    A streaming response chunk.
    
    Attributes:
        session_id: The session ID
        delta: The text delta
        finished: Whether this is the final chunk
    """
    session_id: str
    delta: str
    finished: bool = False

class SessionInfo(BaseModel):
    """
    Information about a chat session.
    
    Attributes:
        session_id: The session ID
        created_at: When the session was created
        last_active: When the session was last active
        message_count: The number of messages in the session
    """
    session_id: str
    created_at: datetime
    last_active: datetime
    message_count: int
