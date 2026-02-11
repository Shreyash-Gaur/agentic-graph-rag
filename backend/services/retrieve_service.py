# backend/services/retrieve_service.py
"""
RetrieveService
- Wraps a FAISS index + metadata JSONL (Legacy/Local)
- Integrates GraphService for Hybrid RAG (Structured + Unstructured)
- Uses EmbedCacheService to avoid re-embedding identical texts
- Falls back to repo Embedder if cache misses
- Optional reranker object can be provided
"""

from __future__ import annotations
import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss

# absolute imports
from backend.services.embed_cache_service import EmbedCacheService
from backend.core.config import settings
from backend.tools.embedder import Embedder

# Import the new GraphService
# We use a try-except block or check settings to ensure it doesn't crash if the file isn't there yet,
# but assuming you've followed the guide, this import is standard.
try:
    from backend.services.graph_service import GraphService
except ImportError:
    GraphService = None

logger = logging.getLogger("agentic-rag.retrieve")

# Default paths (from config if available)
DEFAULT_INDEX = settings.FAISS_INDEX_PATH if hasattr(settings, "FAISS_INDEX_PATH") else "backend/db/vector_data/knowledge_faiss.index"
DEFAULT_META = settings.FAISS_META_PATH if hasattr(settings, "FAISS_META_PATH") else "backend/db/vector_data/knowledge_meta.jsonl"
DEFAULT_EMBED_MODEL = settings.EMBEDDING_MODEL if hasattr(settings, "EMBEDDING_MODEL") else "mxbai-embed-large:latest"


def _load_meta_lines(meta_path: str) -> List[Dict[str, Any]]:
    meta = []
    if not os.path.exists(meta_path):
        return meta
    with open(meta_path, "r", encoding="utf8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                meta.append(json.loads(ln))
            except Exception:
                # tolerate non-json lines
                meta.append({"text": ln})
    return meta


class RetrieveService:
    def __init__(
        self,
        index_path: str = DEFAULT_INDEX,
        meta_path: str = DEFAULT_META,
        embed_cache: Optional[EmbedCacheService] = None,
        embedder: Optional[Embedder] = None,
        reranker_obj: Optional[Any] = None,
        reranker_enabled: bool = False,
        graph_service: Optional[GraphService] = None,  # Injected GraphService
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.reranker = reranker_obj
        self.reranker_enabled = bool(reranker_enabled)

        # 1. Load metadata (Legacy FAISS)
        self.meta: List[Dict[str, Any]] = _load_meta_lines(self.meta_path)
        logger.info("Loaded metadata entries: %d", len(self.meta))

        # 2. Load FAISS index lazily
        self._index: Optional[faiss.Index] = None
        self._index_ntotal: int = 0

        # 3. Embedder / Cache
        self.embedder = embedder or Embedder()
        self.embed_model = getattr(self.embedder, "model", DEFAULT_EMBED_MODEL)
        self.embed_cache = embed_cache or EmbedCacheService()

        # 4. Check FAISS index existence
        if os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
                self._index_ntotal = int(self._index.ntotal)
                logger.info("FAISS index loaded. ntotal=%d", self._index_ntotal)
            except Exception as e:
                logger.exception("Failed to load FAISS index: %s", e)
                self._index = None
        else:
            logger.warning("FAISS index not found at %s", self.index_path)
            self._index = None

        # 5. Initialize Graph Integration
        self.graph_service = graph_service
        self.neo4j_vector = None
        
        if self.graph_service:
            try:
                # We get the vector store interface from the service
                self.neo4j_vector = self.graph_service.get_vector_index()
                logger.info("RetrieveService successfully connected to Neo4j Vector Store.")
            except Exception as e:
                logger.error(f"Failed to attach Neo4j Vector Store: {e}")

    # --------------------------
    # Hybrid Retrieval (Graph + Vector)
    # --------------------------
    def retrieve_hybrid(self, query: str, top_k: int = 5) -> List[str]:
        """
        Combines Graph Structured Data (Cypher) + Vector Search (Neo4j or FAISS).
        Returns a list of strings (context chunks).
        """
        docs = []
        
        # 1. Get Graph Relationships (Structured)
        if self.graph_service:
            try:
                graph_context = self.graph_service.structured_retriever(query)
                if graph_context and graph_context.strip():
                    docs.append(f"**Graph Relationships (Structured Context):**\n{graph_context}")
            except Exception as e:
                logger.error(f"Graph structured retrieval failed: {e}")

        # 2. Get Vector Data (Unstructured)
        # Determine how many candidates to fetch (20 if reranking, 5 if not)
        fetch_k = getattr(settings, "RERANKER_INITIAL_K", 20) if(self.reranker_enabled and self.reranker) else top_k
        
        vector_candidates = []
        used_source = "None"

        # A. Try Neo4j Vector First
        if self.neo4j_vector:
            try:
                results = self.neo4j_vector.similarity_search(query, k=fetch_k)
                vector_candidates = [d.page_content for d in results]
                if vector_candidates:
                    used_source = "Neo4j"
            except Exception as e:
                logger.error(f"Neo4j vector retrieval failed: {e}")

        # B. Fallback to FAISS if Neo4j returned nothing
        if not vector_candidates and self._index is not None:
            logger.info("Neo4j vector empty/unavailable, falling back to FAISS.")
            used_source = "FAISS"
            
            # Embed and search
            qvec = self.embed_text(query).astype("float32")
            D, I = self.search_vector(qvec, fetch_k)
            
            if I.size > 0:
                for idx in I[0]:
                    if idx < 0: continue
                    # Get text from metadata
                    meta = self.meta[idx] if 0 <= idx < len(self.meta) else {}
                    text = meta.get('text', '')
                    if text:
                        vector_candidates.append(text)

        # 3. RERANKING STEP (FIXED)
        if self.reranker_enabled and self.reranker and vector_candidates:
            try:
                print(f"\n--- [RERANKER] Scoring {len(vector_candidates)} docs from {used_source} ---")
                
                # FIX: Use .score() instead of .predict()
                # The wrapper class takes (query, [list of strings])
                scores = self.reranker.score(query, vector_candidates)
                
                # Zip text with scores and sort
                scored_docs = sorted(
                    zip(vector_candidates, scores), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                print(f"--- [RERANKER] Top Score: {scored_docs[0][1]:.4f} ---")
                
                # Keep top_k
                final_vectors = [doc for doc, score in scored_docs[:top_k]]
                docs.extend(final_vectors)
                
            except Exception as e:
                logger.error(f"Hybrid Reranking failed: {e}")
                # Fallback: just take the top k un-reranked
                docs.extend(vector_candidates[:top_k])
        else:
            # No reranker, just take top_k
            docs.extend(vector_candidates[:top_k])

        return docs

    # --------------------------
    # Embedding helpers (with cache)
    # --------------------------
    def embed_text(self, text: str) -> np.ndarray:
        """
        Return a single embedding vector, using cache if available.
        """
        vec = self.embed_cache.get_vector(text, self.embed_model)
        if vec is not None:
            return vec.astype("float32")
        vec = self.embedder.embed_batch([text])[0]
        try:
            self.embed_cache.set_vector(text, self.embed_model, vec)
        except Exception:
            logger.exception("Failed to write embed cache")
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch embedding using cache where possible. Returns np.ndarray shape (n, dim)
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        cached_vecs, keys = self.embed_cache.get_batch(texts, self.embed_model)
        miss_idx = [i for i, v in enumerate(cached_vecs) if v is None]
        results = [v if v is None else np.asarray(v, dtype=np.float32) for v in cached_vecs]

        if miss_idx:
            to_embed = [texts[i] for i in miss_idx]
            try:
                embedded = self.embedder.embed_batch(to_embed)
                embedded = [np.asarray(v, dtype=np.float32) for v in embedded]
            except Exception:
                embedded = [self.embedder.embed(t) for t in to_embed]
                embedded = [np.asarray(v, dtype=np.float32) for v in embedded]

            for idx_local, vec in enumerate(embedded):
                i_global = miss_idx[idx_local]
                results[i_global] = vec
            try:
                self.embed_cache.set_batch(to_embed, self.embed_model, embedded)
            except Exception:
                logger.exception("Failed to set batch in embed cache")

        stacked = np.vstack([np.asarray(v, dtype=np.float32) for v in results])
        return stacked

    # --------------------------
    # Low-level index search (FAISS)
    # --------------------------
    def _ensure_index(self):
        if self._index is None:
            if os.path.exists(self.index_path):
                try:
                    self._index = faiss.read_index(self.index_path)
                    self._index_ntotal = int(self._index.ntotal)
                except Exception as e:
                    logger.exception("Failed to load FAISS index lazily: %s", e)
                    self._index = None

    def search_vector(self, vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query FAISS index with a single vector. Returns distances and indices arrays.
        """
        self._ensure_index()
        if self._index is None:
            # If no index exists, return empty results
            return np.array([[]]), np.array([[]])
            
        q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        D, I = self._index.search(q, top_k)
        return D, I

    # --------------------------
    # High-level retrieve APIs (Standard FAISS)
    # --------------------------
    def _build_result_record(self, idx: int, score: float) -> Dict[str, Any]:
        meta = self.meta[idx] if 0 <= idx < len(self.meta) else {}
        return {"index": idx, "score": float(score), "meta": meta}

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Legacy retrieval: FAISS + Reranker.
        Returns dictionaries with scores and metadata.
        """
        # embed query
        qvec = self.embed_text(query).astype("float32")

        # if reranker enabled, fetch initial candidates larger than top_k
        if self.reranker_enabled and self.reranker:
            initial_k = max(getattr(settings, "RERANKER_INITIAL_K", 20), top_k)
            D, I = self.search_vector(qvec, initial_k)
            candidates = []
            if I.size > 0:
                for dist, idx in zip(D[0], I[0]):
                    if idx < 0:
                        continue
                    candidates.append(self._build_result_record(int(idx), float(dist)))
            
            try:
                print(f"\n--- [RERANKER] Scoring {len(candidates)} documents for query: '{query}' ---") 
    
                reranked = self.reranker.rerank(query, candidates, top_k=top_k)
                if reranked:
                     print(f"--- [RERANKER] Top Doc Score: {reranked[0].get('score', 0):.4f} ---")
                return reranked
            except Exception:
                logger.exception("Reranker failed, falling back to original scores")
                return candidates[:top_k]
        else:
            D, I = self.search_vector(qvec, top_k)
            results = []
            if I.size > 0:
                for dist, idx in zip(D[0], I[0]):
                    if idx < 0:
                        continue
                    results.append(self._build_result_record(int(idx), float(dist)))
            return results

    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieval: embed queries in batch (uses cache) and search each vector.
        """
        if not queries:
            return [[] for _ in queries]

        vecs = self.embed_batch(queries)
        out = []
        for v in vecs:
            D, I = self.search_vector(v, top_k)
            res = []
            if I.size > 0:
                for dist, idx in zip(D[0], I[0]):
                    if idx < 0:
                        continue
                    res.append(self._build_result_record(int(idx), float(dist)))
            out.append(res)
        return out

    # --------------------------
    # Utilities
    # --------------------------
    def reload_meta(self):
        self.meta = _load_meta_lines(self.meta_path)

    def index_count(self) -> int:
        return self._index_ntotal

    def close(self):
        try:
            self.embed_cache.close()
        except Exception:
            pass