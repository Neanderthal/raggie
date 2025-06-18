"""Tests for document reader module."""

import pytest
from unittest.mock import patch, MagicMock
from model_app.core.document_reader import (
    read_pdf_file,
    read_markdown_file,
    read_docx_file,
    read_document_file,
)


class TestFileReaders:
    """Test suite for document reader functions."""
    
    @patch("model_app.core.document_reader.pypdf.PdfReader")
    def test_read_pdf_file(self, mock_reader):
        """Test PDF reading with mocked PdfReader."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "pdf content"
        mock_reader.return_value.pages = [mock_page]

        text = read_pdf_file("dummy.pdf")
        assert "pdf content" in text
        mock_reader.assert_called_once_with("dummy.pdf")

    @patch("model_app.core.document_reader.Path")
    @patch("builtins.open")
    def test_read_markdown_file(self, mock_open, mock_path):
        """Test markdown file reading."""
        mock_path.return_value.suffix = ".md"
        mock_file = MagicMock()
        mock_file.read.return_value = "md content"
        mock_open.return_value.__enter__.return_value = mock_file

        text = read_markdown_file("dummy.md")
        assert text == "md content"
        mock_open.assert_called_once()

    @patch("model_app.core.document_reader.Document")
    def test_read_docx_file(self, mock_doc):
        """Test docx file reading."""
        mock_para = MagicMock()
        mock_para.text = "docx content"
        mock_doc.return_value.paragraphs = [mock_para]

        text = read_docx_file("dummy.docx")
        assert "docx content" in text

    def test_read_unsupported_file(self):
        """Test unsupported file format error."""
        with pytest.raises(ValueError):
            read_document_file("dummy.unsupported")
