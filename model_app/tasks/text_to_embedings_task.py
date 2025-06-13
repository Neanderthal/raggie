import os
import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_experimental.text_splitter import SemanticChunker

from celery_app import celery_app

logger = logging.getLogger(__name__)


# Configuration
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "tokenizer-model")
embedding_url = os.getenv("EMBEDDING_MODEL_URL", "http://localhost:8001/v1")


@celery_app.task(
    bind=True,
    name="model_app.tasks.text_to_embeddings",
    queue="embeddings_queue"
)
def texts_to_embeddings(
    texts: list[str],
    username: str,
    scope_name: str,
    model_name: str = embedding_model_name,
):
    logger.info(f"Starting embeddings task for user {username}, scope {scope_name}")
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
    logger.info(f"Splitting {len(texts)} texts into documents")
    docs = splitter.create_documents(texts)
    logger.info(f"Created {len(docs)} documents from {len(texts)} input texts")

    # Connect to pgvector
    connection_str = f"postgresql+psycopg://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'postgres')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'vector_test')}"
    logger.info(f"Connecting to database at {os.getenv('DB_HOST', 'localhost')}")
    vectorstore = PGVector(
        collection_name="rag_docs",
        connection=connection_str,
        embeddings=embedding_model,
        use_jsonb=True,
    )

    # Store embeddings
    logger.info(f"Storing {len(docs)} documents in vector store")
    try:
        vectorstore.add_documents(docs)
        logger.info("Successfully stored documents in vector store")
    except Exception as e:
        logger.error(f"Failed to store documents: {str(e)}")
        raise
