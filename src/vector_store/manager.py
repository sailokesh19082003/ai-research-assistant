"""
Vector database indexing + retrieval layer.

- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free, no API key
  required) when it can be downloaded from HuggingFace Hub. If the runtime
  has no internet access to huggingface.co, this module automatically falls
  back to a deterministic scikit-learn HashingVectorizer-based embedder so
  the whole pipeline keeps working offline (swap back in automatically the
  moment the real model becomes reachable/cached).
- Store: ChromaDB (persistent, local).
- Search modes: semantic (dense cosine similarity), keyword (BM25 sparse),
  and hybrid (weighted combination of both), matching the spec.
"""
from functools import lru_cache
from typing import List, Dict, Any, Optional

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

from config.settings import settings

EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2's output dimension


class _HashingFallbackEmbedder:
    """Offline, dependency-light stand-in for SentenceTransformer.

    Uses scikit-learn's HashingVectorizer to produce fixed-dimension,
    L2-normalized dense vectors with no network calls and no fitted
    vocabulary (so it's fully consistent across incremental document
    uploads). Not as semantically rich as a real transformer embedding,
    but keeps semantic/hybrid search functional end-to-end when the real
    embedding model can't be downloaded.
    """

    def __init__(self, n_features: int = EMBEDDING_DIM):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._vectorizer = HashingVectorizer(
            n_features=n_features, norm="l2", alternate_sign=False
        )

    def encode(self, texts: List[str], show_progress_bar: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        return self._vectorizer.transform(texts).toarray()


@lru_cache(maxsize=1)
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("[vector_store] Loaded sentence-transformers embedding model.")
        return model
    except Exception as e:
        print(f"[vector_store] Falling back to offline HashingVectorizer embedder "
              f"(sentence-transformers unavailable: {e.__class__.__name__}).")
        return _HashingFallbackEmbedder()


class VectorStoreManager:
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or settings.VECTOR_DB_DIR
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
        self.embedder = get_embedding_model()

        # In-memory BM25 index for keyword/hybrid search. Rebuilt from the
        # Chroma collection lazily whenever it's stale.
        self._bm25 = None
        self._bm25_corpus_ids: List[str] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "file_name": c.get("file_name", ""),
                "page_number": c.get("page_number", -1),
            }
            for c in chunks
        ]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )
        self._bm25 = None  # invalidate cached BM25 index

    def delete_document(self, doc_id: str):
        self.collection.delete(where={"doc_id": doc_id})
        self._bm25 = None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _rebuild_bm25(self):
        all_data = self.collection.get(include=["documents", "metadatas"])
        corpus = all_data.get("documents", []) or []
        ids = all_data.get("ids", []) or []
        tokenized = [doc.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._bm25_corpus_ids = ids
        self._bm25_corpus = corpus
        self._bm25_metadatas = all_data.get("metadatas", []) or []

    def semantic_search(self, query: str, k: int = None, doc_ids: Optional[List[str]] = None):
        k = k or settings.TOP_K
        query_embedding = self.embedder.encode([query]).tolist()

        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=k, where=where
        )
        return self._format_results(results)

    def keyword_search(self, query: str, k: int = None, doc_ids: Optional[List[str]] = None):
        k = k or settings.TOP_K
        if self._bm25 is None:
            self._rebuild_bm25()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(
            zip(self._bm25_corpus_ids, scores, self._bm25_corpus, self._bm25_metadatas),
            key=lambda x: x[1], reverse=True,
        )
        if doc_ids:
            ranked = [r for r in ranked if r[3].get("doc_id") in doc_ids]

        return [
            {"chunk_id": cid, "text": text, "metadata": meta, "score": float(score)}
            for cid, score, text, meta in ranked[:k]
        ]

    def hybrid_search(self, query: str, k: int = None, doc_ids: Optional[List[str]] = None,
                       alpha: float = 0.6):
        """Combines dense semantic similarity (weight=alpha) with sparse BM25
        keyword relevance (weight=1-alpha) for balanced recall/precision."""
        k = k or settings.TOP_K
        dense = self.semantic_search(query, k=k * 3, doc_ids=doc_ids)
        sparse = self.keyword_search(query, k=k * 3, doc_ids=doc_ids)

        def normalize(results, score_key):
            if not results:
                return {}
            scores = [r[score_key] for r in results]
            lo, hi = min(scores), max(scores)
            span = (hi - lo) or 1.0
            return {r["chunk_id"]: (r[score_key] - lo) / span for r in results}

        dense_norm = normalize(dense, "score")
        sparse_norm = normalize(sparse, "score")
        all_ids = set(dense_norm) | set(sparse_norm)

        lookup = {r["chunk_id"]: r for r in dense + sparse}
        combined = []
        for cid in all_ids:
            score = alpha * dense_norm.get(cid, 0.0) + (1 - alpha) * sparse_norm.get(cid, 0.0)
            item = dict(lookup[cid])
            item["score"] = score
            combined.append(item)

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:k]

    @staticmethod
    def _format_results(chroma_results) -> List[Dict[str, Any]]:
        formatted = []
        ids = chroma_results.get("ids", [[]])[0]
        docs = chroma_results.get("documents", [[]])[0]
        metas = chroma_results.get("metadatas", [[]])[0]
        dists = chroma_results.get("distances", [[]])[0]

        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            similarity = 1 - dist  # cosine distance -> similarity
            formatted.append({
                "chunk_id": cid, "text": text, "metadata": meta, "score": similarity
            })
        return formatted


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()
