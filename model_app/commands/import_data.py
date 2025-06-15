import os
import logging
from typing import List
from bs4 import BeautifulSoup
from pathlib import Path
from celery import Celery
from model_app.core.document_reader import (
    read_pdf_file,
    read_markdown_file,
    read_docx_file,
)
from model_app.core.rag import chunk_text


app = Celery(
    "model_app.tasks",
    broker=f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}:5672//",
    backend="rpc://",
    task_default_queue="embeddings_queue",
)

logger = logging.getLogger(__name__)


def _log_file_stats(file_path: str, chunks: List[str]) -> None:
    """Helper to log file processing statistics"""
    try:
        file_size = os.path.getsize(file_path) / 1024  # KB
        avg_chunk_len = sum(len(c) for c in chunks) / max(1, len(chunks))
        logger.debug(
            f"File stats - Path: {file_path}, "
            f"Size: {file_size:.2f}KB, "
            f"Chunks: {len(chunks)}, "
            f"Avg chunk length: {avg_chunk_len:.1f} chars"
        )
    except Exception as e:
        logger.warning(f"Could not calculate file stats for {file_path}: {str(e)}")


def process_file(file_path: str) -> List[str]:
    """Process different file types to extract text chunks"""
    file_extension = Path(file_path).suffix.lower()

    logger.debug(f"Processing file with extension: {file_extension}")
    try:
        if file_extension == ".pdf":
            chunks = chunk_text(read_pdf_file(file_path))
        elif file_extension == ".md":
            chunks = chunk_text(read_markdown_file(file_path))
        elif file_extension in [".docx", ".doc"]:
            if file_extension == ".doc":
                raise ValueError(
                    "Legacy .doc files are not supported. Please convert to .docx format."
                )
            chunks = chunk_text(read_docx_file(file_path))
        elif file_extension == ".html":
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                chunks = chunk_text(soup.get_text())
        elif file_extension == ".txt":
            encodings = ["utf-8", "windows-1251", "cp1251"]
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        chunks = chunk_text(f.read())
                        break
                except UnicodeDecodeError:
                    continue
            else:
                chunks = []
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        _log_file_stats(file_path, chunks)
        return chunks
    except ValueError as e:
        # New + Non-Recoverable error (invalid input format)
        logger.error(f"Validation error processing {file_path}: {str(e)}")
        raise
    except IOError as e:
        # Bubbled-up + Possibly Recoverable error
        logger.error(f"File access error for {file_path}: {str(e)}")
        raise
    except Exception as e:
        # Bubbled-up + Non-Recoverable (unknown error)
        logger.exception("Unexpected error processing file %s", file_path)
        raise RuntimeError(
            f"Failed to process {file_path}: internal error"
        ) from e  # Maintain original error context


def import_data(data_source, username: str, scope_name: str):
    """Import data with user and scope metadata"""
    logger.info(
        f"Starting data import from {data_source} for user {username}, scope {scope_name}"
    )

    if os.path.isdir(data_source):
        file_paths = []
        for root, _, files in os.walk(data_source):
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file_path).suffix.lower() in [
                    ".pdf",
                    ".md",
                    ".docx",
                    ".html",
                    ".txt",
                ]:
                    file_paths.append(file_path)

        logger.info(
            f"Found {len(file_paths)} supported files to process in directory {data_source}\n"
            f"File types: {', '.join(sorted({Path(fp).suffix.lower() for fp in file_paths}))}"
        )
        # Process all files and flatten the list of chunks
        texts = []
        for fp in file_paths:
            try:
                logger.info(f"Processing file: {fp}")
                chunks = process_file(fp)
                texts.extend(chunks)
                logger.info(f"Processed {len(chunks)} chunks from {fp}")

                app.send_task(
                    "model_app.tasks.text_to_embeddings",
                    args=(texts, username, scope_name),  # Changed to tuple
                    kwargs={},  # Explicit empty kwargs
                    queue="embeddings_queue",
                )
                logger.info(f"Sent {len(texts)} total chunks to Celery for embedding")
            except (ValueError, IOError) as e:
                logger.error(f"Specific error processing {fp}: {str(e)}")
                continue
            except Exception as e:
                logger.exception(f"Unexpected error processing {fp}")
                raise RuntimeError("Critical error - aborting import") from e
    else:
        logger.info(f"Processing single file: {data_source}")
        texts = process_file(data_source)
        logger.info(f"Processed {len(texts)} chunks from {data_source}")
        app.send_task(
            "model_app.tasks.text_to_embeddings",
            args=(texts, username, scope_name),  # Changed to tuple
            kwargs={},  # Explicit empty kwargs
            queue="embeddings_queue",
        )
        logger.info(f"Sent {len(texts)} chunks to Celery for embedding")
