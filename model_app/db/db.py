from __future__ import annotations
import os
from typing import Tuple, List, Optional, Dict, Any
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Define SQLModel models
class User(SQLModel, table=True):
    # Use __tablename__ class variable directly
    __tablename__ = "users"  # type: ignore
    
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: Optional[str] = None

class Scope(SQLModel, table=True):
    # Use __tablename__ class variable directly
    __tablename__ = "scopes"  # type: ignore
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None

class Document(SQLModel, table=True):
    # Use __tablename__ class variable directly
    __tablename__ = "documents"  # type: ignore
    
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    # Use SQLAlchemy Column directly for Vector type
    embedding: List[float] = Field(sa_column=Column(Vector(768)))
    user_id: int = Field(foreign_key="users.id")
    scope_id: int = Field(foreign_key="scopes.id")

# Database connection
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL)

def get_session():
    with Session(engine) as session:
        yield session


def get_or_create_user(username: str, email: str | None = None) -> int:
    """Get or create user and return user_id"""
    with Session(engine) as session:
        try:
            # Try to find existing user
            statement = select(User).where(User.username == username)
            user = session.exec(statement).first()
            
            if user:
                # Update email if provided
                if email and user.email != email:
                    user.email = email
                    session.add(user)
                    session.commit()
                # Ensure we return an int, not None
                return user.id if user.id is not None else 0
            
            # Create new user
            new_user = User(username=username, email=email)
            session.add(new_user)
            session.commit()
            session.refresh(new_user)
            return new_user.id if new_user.id is not None else 0
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error creating/getting user: {str(e)}")


def get_or_create_scope(name: str, description: str | None = None) -> int:
    """Get or create scope and return scope_id"""
    with Session(engine) as session:
        try:
            # Try to find existing scope
            statement = select(Scope).where(Scope.name == name)
            scope = session.exec(statement).first()
            
            if scope:
                # Update description if provided
                if description and scope.description != description:
                    scope.description = description
                    session.add(scope)
                    session.commit()
                # Ensure we return an int, not None
                return scope.id if scope.id is not None else 0
            
            # Create new scope
            new_scope = Scope(name=name, description=description)
            session.add(new_scope)
            session.commit()
            session.refresh(new_scope)
            return new_scope.id if new_scope.id is not None else 0
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error creating/getting scope: {str(e)}")


def get_scope_by_name(scope_name: str) -> int:
    with Session(engine) as session:
        statement = select(Scope).where(Scope.name == scope_name)
        scope = session.exec(statement).first()
        # Ensure we return an int, not None
        return scope.id if scope and scope.id is not None else 0


def insert_texts(
    embeddings: list[tuple[str, list[float]]], 
    username: str, 
    scope_name: str
) -> None:
    with Session(engine) as session:
        try:
            user_id = get_or_create_user(username)
            scope_id = get_or_create_scope(scope_name)

            # Create document objects
            documents = [
                Document(content=content, embedding=embedding, user_id=user_id, scope_id=scope_id)
                for content, embedding in embeddings
            ]
            
            # Add all documents
            session.add_all(documents)
            session.commit()
            
            print(f"Imported {len(documents)} documents for user {username} in scope {scope_name}")
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error inserting texts: {str(e)}")


def get_rag_documents(
    scope: str | None, 
    user: str | None, 
    query_embedding: list[float],
    limit: int = 10
) -> list[tuple[str, list[float], float]]:
    with Session(engine) as session:
        try:
            # Start with base query
            query = select(Document.content, Document.embedding)
            
            # Add filters
            if scope:
                scope_id = get_scope_by_name(scope)
                if not scope_id:
                    raise ValueError(f"Scope '{scope}' not found")
                query = query.where(Document.scope_id == scope_id)
                
            if user:
                # Get user ID
                user_statement = select(User).where(User.username == user)
                user_obj = session.exec(user_statement).first()
                if not user_obj:
                    raise ValueError(f"User '{user}' not found")
                query = query.where(Document.user_id == user_obj.id)
            
            # Execute query
            results = session.exec(query).all()
            
            # Calculate similarity scores (this would be more efficient with a raw SQL query,
            # but we're using SQLModel's ORM approach here)
            scored_results = []
            for content, embedding in results:
                # Calculate cosine similarity (simplified)
                # In a real implementation, you'd use a proper vector similarity function
                similarity = 0.5  # Placeholder - in production use proper vector similarity
                scored_results.append((content, embedding, similarity))
            
            # Sort by similarity and limit
            scored_results.sort(key=lambda x: x[2], reverse=True)
            return scored_results[:limit]
            
        except Exception as e:
            raise ValueError(f"Error getting RAG documents: {str(e)}")
