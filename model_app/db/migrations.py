import logging
import sys
from pathlib import Path
from sqlmodel import SQLModel, text

# Add the project root to Python path so imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification (noqa: E402)
from model_app.db.db import engine  # noqa: E402

logger = logging.getLogger(__name__)

def create_pgvector_extension():
    """Create the pgvector extension if it doesn't exist"""
    try:
        with engine.connect() as conn:
            # Check if extension exists
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            if not result.fetchone():
                logger.info("Creating pgvector extension...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension created successfully")
            else:
                logger.info("pgvector extension already exists")
    except Exception as e:
        logger.error(f"Failed to create pgvector extension: {e}")
        raise

def create_langchain_pgvector_tables():
    """Create the tables that langchain-postgres expects"""
    try:
        with engine.connect() as conn:
            # Create the langchain_pg_collection table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR NOT NULL,
                    cmetadata JSON,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            
            # Create the langchain_pg_embedding table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
                    embedding VECTOR(768),
                    document VARCHAR,
                    cmetadata JSON,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            
            # Create indexes for better performance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_collection_id 
                ON langchain_pg_embedding(collection_id)
            """))
            
            # Note: Vector index will be created automatically by langchain-postgres
            # when documents are first added, as it needs to know the vector dimensions
            
            conn.commit()
            logger.info("LangChain PGVector tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create LangChain PGVector tables: {e}")
        raise

def create_all_tables():
    """Create all database tables"""
    try:
        # First create the pgvector extension
        create_pgvector_extension()
        
        # Create SQLModel tables
        logger.info("Creating SQLModel tables...")
        SQLModel.metadata.create_all(engine)
        logger.info("SQLModel tables created successfully")
        
        # Create LangChain PGVector tables
        create_langchain_pgvector_tables()
        
        logger.info("All database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise

def drop_all_tables():
    """Drop all database tables (use with caution!)"""
    try:
        logger.warning("Dropping all database tables...")
        
        # Drop LangChain tables first (due to foreign key constraints)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE"))
            conn.commit()
        
        # Drop SQLModel tables
        SQLModel.metadata.drop_all(engine)
        
        logger.info("All database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise

def reset_database():
    """Reset the entire database (drop and recreate all tables)"""
    logger.warning("Resetting database - this will delete all data!")
    drop_all_tables()
    create_all_tables()
    logger.info("Database reset completed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python migrations.py [create|drop|reset]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        create_all_tables()
    elif command == "drop":
        drop_all_tables()
    elif command == "reset":
        reset_database()
    else:
        print("Unknown command. Use: create, drop, or reset")
        sys.exit(1)
