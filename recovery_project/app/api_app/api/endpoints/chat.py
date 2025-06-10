import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api_app.api.deps import get_chat_service, get_api_key
from api_app.schemas.chat_schemas import ChatRequest, ChatResponse, SessionInfo
from api_app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    api_key: str = Depends(get_api_key),
) -> ChatResponse:
    """
    Process a chat request and return a response.
    
    This endpoint:
    - Accepts a chat request with messages and optional filters
    - Retrieves relevant documents using RAG
    - Generates a response using the AI model
    - Returns the response
    """
    try:
        return await chat_service.chat(req)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}",
        )

@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    api_key: str = Depends(get_api_key),
):
    """
    Process a chat request and stream the response.
    
    This endpoint:
    - Accepts a chat request with messages and optional filters
    - Retrieves relevant documents using RAG
    - Streams the response from the AI model
    - Returns a streaming response
    """
    try:
        async def stream_generator():
            async for chunk in chat_service.stream_chat(req):
                # Format as SSE
                data = f"data: {chunk.model_dump_json()}\n\n"
                yield data
            
            # End of stream marker
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing streaming chat request: {str(e)}",
        )

@router.get("/sessions", response_model=List[SessionInfo])
async def list_sessions(
    chat_service: ChatService = Depends(get_chat_service),
    api_key: str = Depends(get_api_key),
) -> List[SessionInfo]:
    """
    List all active chat sessions.
    
    This endpoint:
    - Returns a list of all active chat sessions
    - Includes metadata such as creation time and message count
    """
    try:
        sessions_dict = chat_service.list_sessions()
        # Convert dictionary to list of SessionInfo objects
        return [
            SessionInfo(
                session_id=session_id,
                created_at=info["created_at"],
                last_active=info["last_active"],
                message_count=info["message_count"]
            )
            for session_id, info in sessions_dict.items()
        ]
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing sessions: {str(e)}",
        )

@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),
    api_key: str = Depends(get_api_key),
) -> SessionInfo:
    """
    Get information about a specific chat session.
    
    This endpoint:
    - Returns metadata for a specific chat session
    - Includes creation time and message count
    """
    try:
        session_info = chat_service.get_session_info(session_id)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
        # Convert dictionary to SessionInfo object
        return SessionInfo(
            session_id=session_id,
            created_at=session_info["created_at"],
            last_active=session_info["last_active"],
            message_count=session_info["message_count"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting session: {str(e)}",
        )

@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),
    api_key: str = Depends(get_api_key),
):
    """
    Clear a chat session's history.
    
    This endpoint:
    - Deletes all messages for a specific chat session
    - Returns 204 No Content on success
    """
    try:
        # Check if session exists first
        session_info = chat_service.get_session_info(session_id)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
        
        # Clear the session
        chat_service.clear_session(session_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing session: {str(e)}",
        )
