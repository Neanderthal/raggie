from __future__ import annotations
import os
from typing import Tuple
import asyncpg
from pgvector.asyncpg import register_vector
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


async def get_connection() -> asyncpg.Connection:
    conn = await asyncpg.connect(
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
    await register_vector(conn)
    return conn


async def get_or_create_user(username: str, email: str | None = None) -> int:
    """Get or create user and return user_id"""
    conn = await get_connection()
    try:
        result = await conn.fetchrow(
            "INSERT INTO users (username, email) VALUES ($1, $2) ON CONFLICT (username) DO UPDATE SET email = EXCLUDED.email RETURNING id",
            username, email,
        )
        if not result:
            raise ValueError(f"Failed to get user ID for {username}")
        return result['id']
    except Exception as e:
        raise ValueError(f"Error creating/getting user: {str(e)}")
    finally:
        await conn.close()


async def get_or_create_scope(name: str, description: str | None = None) -> int:
    """Get or create scope and return scope_id"""
    conn = await get_connection()
    try:
        result = await conn.fetchrow(
            "INSERT INTO scopes (name, description) VALUES ($1, $2) ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description RETURNING id",
            name, description,
        )
        if not result:
            raise ValueError(f"Failed to get scope ID for {name}")
        return result['id']
    except Exception as e:
        raise ValueError(f"Error creating/getting scope: {str(e)}")
    finally:
        await conn.close()


async def get_scope_by_name(scope_name: str) -> Tuple[int] | None:
    conn = await get_connection()
    try:
        return await conn.fetchrow("SELECT id FROM scopes WHERE name = $1", scope_name)
    finally:
        await conn.close()


async def insert_texts(
    embeddings: list[tuple[str, list[float]]], 
    username: str, 
    scope_name: str
) -> None:
    conn = await get_connection()
    try:
        user_id = await get_or_create_user(username)
        scope_id = await get_or_create_scope(scope_name)

        data = [
            (content, embedding, user_id, scope_id) for (content, embedding) in embeddings
        ]

        await conn.executemany(
            "INSERT INTO documents (content, embedding, user_id, scope_id) VALUES ($1, $2, $3, $4)",
            data,
        )
        print(f"Imported {len(data)} documents for user {username} in scope {scope_name}")
    finally:
        await conn.close()


async def get_rag_documents(
    scope: str | None, 
    user: str | None, 
    query_embedding: list[float]
) -> list[tuple[str, list[float], float]]:

    conn = await get_connection()
    try:
        # Base query with optional filters
        params = []
        params.append(query_embedding)
        
        query_sql = """
            SELECT d.content, d.embedding, (d.embedding <=> $1::vector(768)) as similarity
            FROM documents d
            WHERE 1=1
        """

        if scope:
            scope_result = await get_scope_by_name(scope)
            if not scope_result:
                raise ValueError(f"Scope '{scope}' not found")
            query_sql += " AND d.scope_id = $2"
            params.append(scope_result[0])

        if user:
            user_result = await conn.fetchrow("SELECT id FROM users WHERE username = $1", user)
            if not user_result:
                raise ValueError(f"User '{user}' not found")
            query_sql += " AND d.user_id = $3" if scope else " AND d.user_id = $2"
            params.append(user_result['id'])

        query_sql += " ORDER BY d.embedding <=> $1::vector(768) LIMIT 10"
        return await conn.fetch(query_sql, *params)
    finally:
        await conn.close()
