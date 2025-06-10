import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_experimental.text_splitter import SemanticChunker

from celery_app import celery_app


# Configuration
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "tokenizer-model")
embedding_url = os.getenv("EMBEDDING_MODEL_URL", "http://localhost:8001/v1")


@celery_app.task
def texts_to_embeddings(
    texts: list[str],
    username: str,
    scope_name: str,
    model_name: str = embedding_model_name,
):
    # Embedding model via OpenAI-compatible API
    os.environ["OPENAI_API_BASE"] = embedding_url
    embedding_model = OpenAIEmbeddings(
        model=embedding_model_name
    )
    # Semantic-aware text splitting (fast fallback option)
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # docs = splitter.create_documents(texts)

    # Optionally: use LangChain's SemanticChunker if you want sentence-aware embedding-based grouping
    splitter = SemanticChunker(embedding_model)
    docs = splitter.create_documents(texts)

    # Connect to pgvector
    connection_str = f"postgresql+psycopg://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'postgres')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'vector_test')}"
    vectorstore = PGVector(
        collection_name="rag_docs",
        connection=connection_str,
        embeddings=embedding_model,
        use_jsonb=True,
    )

    # Store embeddings
    vectorstore.add_documents(docs)
