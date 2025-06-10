import logging
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from api_app.services.ai_provider import AIProvider
from api_app.services.db_service import db_service
from api_app.utils.session_manager import session_manager
from api_app.schemas.chat_schemas import ChatRequest, ChatResponse, Message, StreamChunk

logger = logging.getLogger(__name__)

class ChatService:
    """
    Service for handling chat interactions.
    
    This class handles:
    - Processing chat requests
    - Managing chat sessions
    - Integrating RAG capabilities
    - Streaming responses
    """
    
    def __init__(self):
        self.ai_provider = AIProvider()
    
    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Process a chat request and return a response"""
        try:
            # Store the user message in session history
            user_message = req.messages[-1]
            session_manager.append_message(
                req.session_id, 
                {"role": user_message.role, "content": user_message.content}
            )
            
            # Get full conversation history
            history = session_manager.get_history(req.session_id)
            
            # Get relevant documents for RAG if this is a user message
            context_docs = []
            if user_message.role == "user":
                # Generate embedding for the query
                query_embedding = await self.ai_provider.get_embedding(user_message.content)
                
                # Get relevant documents
                rag_results = await db_service.get_rag_documents(
                    scope=req.scope,
                    user=req.username,
                    query_embedding=query_embedding
                )
                
                # Extract just the content from the results
                context_docs = [doc[0] for doc in rag_results]
                logger.info(f"Found {len(context_docs)} relevant documents for RAG")
            
            # Get response from AI model
            response_content = await self.ai_provider.get_chat_response(
                messages=history,
                context_docs=context_docs
            )
            
            # Store the assistant's response in session history
            session_manager.append_message(
                req.session_id,
                {"role": "assistant", "content": response_content}
            )
            
            # Create and return the response
            return ChatResponse(
                message=Message(role="assistant", content=response_content),
                session_id=req.session_id,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}")
            raise
    
    async def stream_chat(self, req: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """Process a chat request and stream the response"""
        try:
            # Store the user message in session history
            user_message = req.messages[-1]
            session_manager.append_message(
                req.session_id, 
                {"role": user_message.role, "content": user_message.content}
            )
            
            # Get full conversation history
            history = session_manager.get_history(req.session_id)
            
            # Get relevant documents for RAG if this is a user message
            context_docs = []
            if user_message.role == "user":
                # Generate embedding for the query
                query_embedding = await self.ai_provider.get_embedding(user_message.content)
                
                # Get relevant documents
                rag_results = await db_service.get_rag_documents(
                    scope=req.scope,
                    user=req.username,
                    query_embedding=query_embedding
                )
                
                # Extract just the content from the results
                context_docs = [doc[0] for doc in rag_results]
                logger.info(f"Found {len(context_docs)} relevant documents for RAG")
            
            # Collect the full response while streaming
            full_response = ""
            
            # Stream response from AI model
            async for chunk in self.ai_provider.stream_chat_response(
                messages=history,
                context_docs=context_docs
            ):
                full_response += chunk
                
                # Yield each chunk
                yield StreamChunk(
                    session_id=req.session_id,
                    delta=chunk,
                    finished=False
                )
            
            # Store the complete assistant's response in session history
            session_manager.append_message(
                req.session_id,
                {"role": "assistant", "content": full_response}
            )
            
            # Yield final chunk to indicate completion
            yield StreamChunk(
                session_id=req.session_id,
                delta="",
                finished=True
            )
            
        except Exception as e:
            logger.error(f"Error streaming chat response: {str(e)}")
            raise
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a chat session"""
        return session_manager.get_session_info(session_id)
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all active chat sessions"""
        return session_manager.get_all_sessions()
    
    def clear_session(self, session_id: str) -> None:
        """Clear a chat session's history"""
        session_manager.clear_session(session_id)
