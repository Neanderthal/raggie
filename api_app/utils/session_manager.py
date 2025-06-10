"""
Session manager for chat sessions.

This module provides a session manager for chat sessions.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manager for chat sessions.
    
    This class provides methods to manage chat sessions, including:
    - Creating sessions
    - Storing and retrieving messages
    - Managing session metadata
    """
    
    def __init__(self):
        """Initialize the session manager."""
        # Dictionary to store sessions
        # Key: session_id, Value: list of messages
        self.sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        
        # Dictionary to store session metadata
        # Key: session_id, Value: session metadata
        self.session_info: Dict[str, Dict[str, Any]] = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Initialized session manager")
        
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get the message history for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            The list of messages for the session
        """
        with self.lock:
            # Create session info if it doesn't exist
            if session_id not in self.session_info:
                self._create_session_info(session_id)
                
            return self.sessions[session_id].copy()
            
    def append_message(self, session_id: str, message: Dict[str, str]) -> None:
        """
        Append a message to a session.
        
        Args:
            session_id: The session ID
            message: The message to append
        """
        with self.lock:
            # Create session info if it doesn't exist
            if session_id not in self.session_info:
                self._create_session_info(session_id)
                
            # Append the message
            self.sessions[session_id].append(message)
            
            # Update session metadata
            self.session_info[session_id]["last_active"] = datetime.now()
            self.session_info[session_id]["message_count"] += 1
            
    def clear_session(self, session_id: str) -> None:
        """
        Clear a session's history.
        
        Args:
            session_id: The session ID
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id] = []
                
                # Reset message count but keep other metadata
                if session_id in self.session_info:
                    self.session_info[session_id]["message_count"] = 0
                    self.session_info[session_id]["last_active"] = datetime.now()
                    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            The session information, or None if the session doesn't exist
        """
        with self.lock:
            return self.session_info.get(session_id)
            
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all sessions.
        
        Returns:
            A dictionary of session information
        """
        with self.lock:
            return self.session_info.copy()
            
    def _create_session_info(self, session_id: str) -> None:
        """
        Create session info for a new session.
        
        Args:
            session_id: The session ID
        """
        now = datetime.now()
        self.session_info[session_id] = {
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }

# Create a global session manager instance
session_manager = SessionManager()
