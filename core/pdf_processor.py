"""
core/pdf_processor.py
Handles PDF ingestion: text extraction → cleaning → chunking → metadata tagging.
Uses pdfplumber for layout-aware extraction and LangChain's RecursiveCharacterTextSplitter.
"""

import re
import hashlib
from pathlib import Path
from typing import List

import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def _clean_text(text: str) -> str:
    """Remove excessive whitespace, hyphenation artefacts, and noise."""
    text = re.sub(r"-\n", "", text)          # rejoin hyphenated line-breaks
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse triple+ newlines
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse multiple spaces
    return text.strip()


def _pdf_hash(file_bytes: bytes) -> str:
    """Stable unique ID for a PDF file based on its content."""
    return hashlib.md5(file_bytes).hexdigest()[:12]


def extract_documents(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Full pipeline: PDF bytes → cleaned LangChain Documents with metadata.

    Each returned Document represents one chunk and carries:
        - source   : original filename
        - page     : page number (1-indexed)
        - doc_id   : stable hash of the PDF
        - chunk_id : unique id for this chunk
    """
    import io
    pdf_id = _pdf_hash(file_bytes)
    raw_pages: List[tuple[int, str]] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = _clean_text(text)
            if text:
                raw_pages.append((page_num, text))

    if not raw_pages:
        return []

    # Build one Document per page first, then chunk
    page_docs = [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": page_num,
                "doc_id": pdf_id,
            },
        )
        for page_num, text in raw_pages
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(page_docs)

    # Add a unique chunk_id to each chunk for dedup / tracing
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{pdf_id}-{i:04d}"

    return chunks


def get_paper_title(file_bytes: bytes) -> str:
    """
    Best-effort title extraction: tries to read the first non-empty line
    of the first page, which is usually the paper title.
    """
    import io
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        if pdf.pages:
            text = pdf.pages[0].extract_text() or ""
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                return lines[0][:120]
    return "Unknown Title"
