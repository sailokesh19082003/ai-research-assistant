"""
Recursive-style text chunking that preserves page-number metadata across
overlapping segments so citations remain accurate after retrieval.
"""
from typing import List, Dict, Any
from config.settings import settings


class DocumentChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def create_chunks(self, pages_data: List[Dict[str, Any]], file_name: str) -> List[Dict[str, Any]]:
        """Splits each page's text into overlapping chunks (~800-1000 chars,
        ~100-150 char overlap) while keeping the originating page number."""
        chunks = []
        chunk_id = 0

        for page in pages_data:
            text = page["text"]
            start = 0
            n = len(text)

            if n == 0:
                continue

            while start < n:
                end = min(start + self.chunk_size, n)
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "chunk_id": f"{page['doc_id']}_c{chunk_id}",
                        "doc_id": page["doc_id"],
                        "file_name": file_name,
                        "page_number": page["page_number"],
                        "text": chunk_text,
                    })
                    chunk_id += 1

                if end == n:
                    break
                start += max(1, self.chunk_size - self.chunk_overlap)

        return chunks
