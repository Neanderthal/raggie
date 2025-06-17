import os
from typing import Optional, List
from datetime import datetime
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import delete
from dotenv import load_dotenv

# Import pgvector for alembic

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


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


class InitialDocument(SQLModel, table=True):
    # Use __tablename__ class variable directly
    __tablename__ = "documents"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    # Use SQLAlchemy Column directly for Vector type
    user_id: int = Field(foreign_key="users.id")
    scope_id: int = Field(foreign_key="scopes.id")


class VectorDocument(SQLModel, table=True):
    __tablename__ = "vector_documents"  # type: ignore
    
    id: Optional[int] = Field(default=None, primary_key=True)
    initial_document_id: int = Field(foreign_key="documents.id")
    vector_store_id: str = Field(index=True)  # UUID from vector store
    chunk_index: int = Field(default=0)  # Order of chunk within the document
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Database connection
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL, echo=False)


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
    embeddings: list[tuple[str, list[float]]], username: str, scope_name: str
) -> None:
    with Session(engine) as session:
        try:
            user_id = get_or_create_user(username)
            scope_id = get_or_create_scope(scope_name)

            # Create document objects
            documents = [
                InitialDocument(
                    content=content,
                    user_id=user_id,
                    scope_id=scope_id,
                )
                for content, embedding in embeddings
            ]

            # Add all documents
            session.add_all(documents)
            session.commit()

            print(
                f"Imported {len(documents)} documents for user {username} in scope {scope_name}"
            )
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error inserting texts: {str(e)}")


def get_rag_documents(
    scope: str | None, user: str | None, query_embedding: list[float], limit: int = 10
) -> list[tuple[str, list[float], float]]:
    with Session(engine) as session:
        try:
            # Start with base query
            query = select(InitialDocument.content)

            # Add filters
            if scope:
                scope_id = get_scope_by_name(scope)
                if not scope_id:
                    raise ValueError(f"Scope '{scope}' not found")
                query = query.where(InitialDocument.scope_id == scope_id)

            if user:
                # Get user ID
                user_statement = select(User).where(User.username == user)
                user_obj = session.exec(user_statement).first()
                if not user_obj:
                    raise ValueError(f"User '{user}' not found")
                query = query.where(InitialDocument.user_id == user_obj.id)

            # Execute query
            results = session.exec(query).all()

            # Calculate similarity scores (this would be more efficient with a raw SQL query,
            # but we're using SQLModel's ORM approach here)
            scored_results = []
            for content, embedding in results:
                # Calculate cosine similarity (simplified)
                # In a real implementation, you'd use a proper vector similarity function
                similarity = (
                    0.5  # Placeholder - in production use proper vector similarity
                )
                scored_results.append((content, embedding, similarity))

            # Sort by similarity and limit
            scored_results.sort(key=lambda x: x[2], reverse=True)
            return scored_results[:limit]

        except Exception as e:
            raise ValueError(f"Error getting RAG documents: {str(e)}")


def create_initial_document(content: str, user_id: int, scope_id: int) -> int:
    """Create an initial document and return its ID."""
    with Session(engine) as session:
        try:
            initial_doc = InitialDocument(
                content=content,
                user_id=user_id,
                scope_id=scope_id,
            )
            session.add(initial_doc)
            session.commit()
            session.refresh(initial_doc)
            return initial_doc.id if initial_doc.id is not None else 0
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error creating initial document: {str(e)}")


def store_vector_document_links(initial_document_id: int, vector_store_ids: List[str]) -> None:
    """Store the relationship between initial document and vector store documents."""
    with Session(engine) as session:
        try:
            for i, vector_id in enumerate(vector_store_ids):
                vector_doc = VectorDocument(
                    initial_document_id=initial_document_id,
                    vector_store_id=vector_id,
                    chunk_index=i
                )
                session.add(vector_doc)
            session.commit()
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error storing vector document links: {str(e)}")


def get_vector_documents_by_initial_id(initial_document_id: int) -> List[str]:
    """Get all vector store document IDs for an initial document."""
    with Session(engine) as session:
        vector_docs = session.exec(
            select(VectorDocument.vector_store_id)
            .where(VectorDocument.initial_document_id == initial_document_id)
            .order_by("chunk_index")
        ).all()
        return list(vector_docs)


def get_initial_document_by_vector_id(vector_store_id: str) -> Optional[int]:
    """Get the initial document ID for a vector store document."""
    with Session(engine) as session:
        vector_doc = session.exec(
            select(VectorDocument.initial_document_id)
            .where(VectorDocument.vector_store_id == vector_store_id)
        ).first()
        return vector_doc


def clear_all_data() -> int:
    """Clear all RAG data from the database. Returns count of deleted documents."""
    with Session(engine) as session:
        try:
            # Count documents before deletion
            try:
                doc_count = session.exec(select(InitialDocument)).all()
                count = len(doc_count)
            except Exception:
                # If InitialDocument table doesn't exist, return 0
                return 0
            
            # Delete in order due to foreign key constraints
            # Handle case where tables might not exist
            try:
                session.execute(delete(VectorDocument))
            except Exception:
                pass  # Table doesn't exist, skip
            
            try:
                session.execute(delete(InitialDocument))
            except Exception:
                pass  # Table doesn't exist, skip
            
            try:
                session.execute(delete(User))
            except Exception:
                pass  # Table doesn't exist, skip
            
            try:
                session.execute(delete(Scope))
            except Exception:
                pass  # Table doesn't exist, skip
            
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error clearing all data: {str(e)}")


def clear_user_data(username: str) -> int:
    """Clear all data for a specific user. Returns count of deleted documents."""
    with Session(engine) as session:
        try:
            # Find user
            try:
                user = session.exec(select(User).where(User.username == username)).first()
                if not user:
                    return 0
            except Exception:
                # User table doesn't exist
                return 0
            
            # Get user's documents
            try:
                user_docs = session.exec(
                    select(InitialDocument).where(InitialDocument.user_id == user.id)
                ).all()
                count = len(user_docs)
            except Exception:
                # InitialDocument table doesn't exist
                count = 0
                user_docs = []
            
            # Delete vector documents for this user's initial documents
            for doc in user_docs:
                if doc.id is not None:
                    doc_id = doc.id
                    try:
                        session.execute(
                            delete(VectorDocument).where(VectorDocument.initial_document_id == doc_id)
                        )
                    except Exception:
                        pass  # VectorDocument table doesn't exist, skip
            
            # Delete user's initial documents
            if user.id is not None:
                user_id = user.id
                try:
                    session.execute(
                        delete(InitialDocument).where(InitialDocument.user_id == user_id)
                    )
                except Exception:
                    pass  # InitialDocument table doesn't exist, skip
            
            # Delete user
            session.delete(user)
            
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error clearing user data: {str(e)}")


def clear_scope_data(scope_name: str) -> int:
    """Clear all data for a specific scope. Returns count of deleted documents."""
    with Session(engine) as session:
        try:
            # Find scope
            try:
                scope = session.exec(select(Scope).where(Scope.name == scope_name)).first()
                if not scope:
                    return 0
            except Exception:
                # Scope table doesn't exist
                return 0
            
            # Get scope's documents
            try:
                scope_docs = session.exec(
                    select(InitialDocument).where(InitialDocument.scope_id == scope.id)
                ).all()
                count = len(scope_docs)
            except Exception:
                # InitialDocument table doesn't exist
                count = 0
                scope_docs = []
            
            # Delete vector documents for this scope's initial documents
            for doc in scope_docs:
                if doc.id is not None:
                    doc_id = doc.id
                    try:
                        session.execute(
                            delete(VectorDocument).where(VectorDocument.initial_document_id == doc_id)
                        )
                    except Exception:
                        pass  # VectorDocument table doesn't exist, skip
            
            # Delete scope's initial documents
            if scope.id is not None:
                scope_id = scope.id
                try:
                    session.execute(
                        delete(InitialDocument).where(InitialDocument.scope_id == scope_id)
                    )
                except Exception:
                    pass  # InitialDocument table doesn't exist, skip
            
            # Delete scope
            session.delete(scope)
            
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error clearing scope data: {str(e)}")


def clear_user_scope_data(username: str, scope_name: str) -> int:
    """Clear data for a specific user and scope combination. Returns count of deleted documents."""
    with Session(engine) as session:
        try:
            # Find user and scope
            try:
                user = session.exec(select(User).where(User.username == username)).first()
                scope = session.exec(select(Scope).where(Scope.name == scope_name)).first()
                
                if not user or not scope:
                    return 0
            except Exception:
                # User or Scope table doesn't exist
                return 0
            
            # Get documents for this user and scope
            try:
                user_scope_docs = session.exec(
                    select(InitialDocument).where(
                        InitialDocument.user_id == user.id,
                        InitialDocument.scope_id == scope.id
                    )
                ).all()
                count = len(user_scope_docs)
            except Exception:
                # InitialDocument table doesn't exist
                count = 0
                user_scope_docs = []
            
            # Delete vector documents for these initial documents
            for doc in user_scope_docs:
                if doc.id is not None:
                    doc_id = doc.id
                    try:
                        session.execute(
                            delete(VectorDocument).where(VectorDocument.initial_document_id == doc_id)
                        )
                    except Exception:
                        pass  # VectorDocument table doesn't exist, skip
            
            # Delete initial documents
            if user.id is not None and scope.id is not None:
                user_id = user.id
                scope_id = scope.id
                try:
                    session.execute(
                        delete(InitialDocument).where(
                            (InitialDocument.user_id == user_id) & (InitialDocument.scope_id == scope_id)
                        )
                    )
                except Exception:
                    pass  # InitialDocument table doesn't exist, skip
            
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error clearing user scope data: {str(e)}")
