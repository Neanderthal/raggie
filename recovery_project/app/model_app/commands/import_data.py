import os
from typing import List
from bs4 import BeautifulSoup
from pathlib import Path
from celery import Celery
from model_app.core.document_reader import (
    read_pdf_file,
    read_markdown_file,
    read_docx_file,
    chunk_text,
)


app = Celery(
    "tasks",
    broker=f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}:5672//",
    backend="rpc://",
)


def process_file(file_path: str) -> List[str]:
    """Process different file types to extract text chunks"""
    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".pdf":
        return chunk_text(read_pdf_file(file_path))
    elif file_extension == ".md":
        return chunk_text(read_markdown_file(file_path))
    elif file_extension in [".docx", ".doc"]:
        if file_extension == ".doc":
            raise ValueError(
                "Legacy .doc files are not supported. Please convert to .docx format."
            )
        return chunk_text(read_docx_file(file_path))
    elif file_extension == ".html":
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            return chunk_text(soup.get_text())
    elif file_extension == ".txt":
        encodings = ["utf-8", "windows-1251", "cp1251"]
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return chunk_text(f.read())
            except UnicodeDecodeError:
                continue
        return []
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def import_data(data_source, username: str, scope_name: str):
    """Import data with user and scope metadata"""

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

        # Process all files and flatten the list of chunks
        texts = []
        for fp in file_paths:
            try:
                texts.extend(process_file(fp))

                app.send_task(
                    "tasks.texts_to_embeddings",
                    args=[texts, username, scope_name],
                )
            except Exception as e:
                print(f"Error processing {fp}: {str(e)}")
                continue
    else:
        texts = process_file(data_source)
        app.send_task(
            "tasks.texts_to_embeddings",
            args=[texts, username, scope_name],
        )
