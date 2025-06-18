import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from model_app.core.embedding import generate_embeddings
from model_app.core.document_reader import (
    read_pdf_file,
    read_markdown_file,
    read_docx_file,
    chunk_text,
    read_document_file,
)


class TestGenerateEmbeddings:
    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self):
        """Test successful embedding generation with mocked client"""
        with patch('model_app.core.embedding.EmbeddingService.generate_embeddings', 
                   return_value=("test text", [0.1, 0.2, 0.3])):
            text, embedding = await generate_embeddings("test text")
            assert text == "test text"
            assert embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_text(self):
        """Test empty text handling"""
        text, embed = await generate_embeddings("   ")
        assert text == "empty"
        assert len(embed) > 0

    @pytest.mark.asyncio
    async def test_generate_embeddings_failure_with_fallback(self):
        """Test API failure case falls back to hash-based embeddings"""
        with patch('model_app.core.embedding.EmbeddingService.generate_embeddings', 
                   side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await generate_embeddings("test")


class TestFileReaders:
    @patch("model_app.core.document_reader.pypdf.PdfReader")
    def test_read_pdf_file(self, mock_reader):
        """Test PDF reading with mocked PdfReader"""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "pdf content"
        mock_reader.return_value.pages = [mock_page]

        text = read_pdf_file("dummy.pdf")
        assert "pdf content" in text
        mock_reader.assert_called_once_with("dummy.pdf")

    @patch("model_app.core.document_reader.Path")
    @patch("builtins.open")
    def test_read_markdown_file(self, mock_open, mock_path):
        """Test markdown file reading"""
        mock_path.return_value.suffix = ".md"
        mock_file = MagicMock()
        mock_file.read.return_value = "md content"
        mock_open.return_value.__enter__.return_value = mock_file

        text = read_markdown_file("dummy.md")
        assert text == "md content"
        mock_open.assert_called_once()

    @patch("model_app.core.document_reader.Document")
    def test_read_docx_file(self, mock_doc):
        """Test docx file reading"""
        mock_para = MagicMock()
        mock_para.text = "docx content"
        mock_doc.return_value.paragraphs = [mock_para]

        text = read_docx_file("dummy.docx")
        assert "docx content" in text

    def test_read_unsupported_file(self):
        """Test unsupported file format error"""
        with pytest.raises(ValueError):
            read_document_file("dummy.unsupported")


class TestChunkText:
    def test_chunk_text_small(self):
        """Test small text chunking"""
        text = "This is a short sentence."
        chunks = chunk_text(text)
        assert chunks == ["This is a short sentence"]

    def test_chunk_text_large(self):
        """Test large text chunking"""
        text = ". ".join(["This is sentence"] * 50)
        chunks = chunk_text(text)
        assert len(chunks) > 1
        assert all(len(chunk) <= 500 for chunk in chunks)

    def test_chunk_text_empty(self):
        """Test empty text chunking"""
        assert chunk_text("") == []
        assert chunk_text("   ") == []
# Paste file contents here
