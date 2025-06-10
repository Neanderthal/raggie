import pypdf
from pathlib import Path
from docx import Document


def read_pdf_file(pdf_path: str | Path) -> str:
    """Read and extract text from PDF file"""
    pdf_document = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page_number in range(len(pdf_document.pages)):
        page = pdf_document.pages[page_number]
        text = page.extract_text()
        full_text += text + " "
    return full_text.strip()


def read_markdown_file(md_path: str | Path) -> str:
    """Read and extract text from markdown file"""
    with open(md_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def read_docx_file(docx_path: str) -> str:
    """Read and extract text from Word document"""
    doc = Document(docx_path)
    full_text = ""
    for paragraph in doc.paragraphs:
        full_text += paragraph.text.strip() + " "
    return full_text.strip()


def read_document_file(file_path: str) -> str:
    """Read document file based on extension"""
    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".pdf":
        return read_pdf_file(file_path)
    elif file_extension == ".md":
        return read_markdown_file(file_path)
    elif file_extension in [".docx", ".doc"]:
        if file_extension == ".doc":
            raise ValueError(
                "Legacy .doc files are not supported. Please convert to .docx format."
            )
        return read_docx_file(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. Supported formats: .pdf, .md, .docx"
        )


def chunk_text(text: str) -> list[str]:
    """Split text into semantic chunks"""
    if not text.strip():
        return []

    chunks = []
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    current_chunk = ""
    for sentence in sentences:
        # Aim for chunks of roughly 200-500 characters
        if len(current_chunk) + len(sentence) < 500:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
