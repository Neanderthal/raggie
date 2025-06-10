import os
import psycopg2
from psycopg2 import sql
from typing import Dict, Any


def create_db():
    # Build connection parameters
    conn_params: Dict[str, Any] = {}
    if os.getenv("DB_USER"):
        conn_params["user"] = os.getenv("DB_USER")
    if os.getenv("DB_PASSWORD"):
        conn_params["password"] = os.getenv("DB_PASSWORD")
    if os.getenv("DB_HOST"):
        conn_params["host"] = os.getenv("DB_HOST")
    if os.getenv("DB_PORT"):
        conn_params["port"] = os.getenv("DB_PORT")

    # Connect without specifying database to check/create it
    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True

    # Create database if it doesn't exist
    db_name = os.getenv("DB_NAME")
    if not db_name:
        raise ValueError("DB_NAME environment variable not set")

    with conn.cursor() as cursor:
        cursor.execute(
            sql.SQL("SELECT 1 FROM pg_database WHERE datname = {}").format(
                sql.Literal(db_name)
            )
        )
        if not cursor.fetchone():
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
            )
            print(f"Database {db_name} created.")

    conn.close()

    # Connect to the target database
    conn_params["dbname"] = db_name
    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True

    with conn.cursor() as cursor:
        # Create vector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                email VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create scopes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scopes (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create documents table with foreign keys
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR(768) NOT NULL,
                user_id INTEGER REFERENCES users(id),
                scope_id INTEGER REFERENCES scopes(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_user 
            ON documents(user_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_scope 
            ON documents(scope_id)
        """
        )

    print("Database schema setup completed.")
    conn.close()
