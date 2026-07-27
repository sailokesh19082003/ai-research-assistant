import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.document_processing.chunker import DocumentChunker


def test_chunker_preserves_page_numbers():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    pages_data = [
        {"doc_id": "doc1", "page_number": 1, "text": "A" * 120},
        {"doc_id": "doc1", "page_number": 2, "text": "B" * 60},
    ]
    chunks = chunker.create_chunks(pages_data, file_name="test.pdf")

    assert len(chunks) > 0
    assert all(c["doc_id"] == "doc1" for c in chunks)
    page1_chunks = [c for c in chunks if c["page_number"] == 1]
    page2_chunks = [c for c in chunks if c["page_number"] == 2]
    assert len(page1_chunks) > 0
    assert len(page2_chunks) > 0


def test_chunker_overlap_behavior():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    pages_data = [{"doc_id": "doc1", "page_number": 1, "text": "X" * 250}]
    chunks = chunker.create_chunks(pages_data, file_name="test.pdf")
    assert len(chunks) >= 3


def test_chunker_handles_empty_page():
    chunker = DocumentChunker()
    chunks = chunker.create_chunks(
        [{"doc_id": "doc1", "page_number": 1, "text": ""}], file_name="test.pdf"
    )
    assert chunks == []
