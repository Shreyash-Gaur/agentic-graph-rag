# backend/services/retrieve_service.py
"""
RetrieveService (Hybrid: Graph + SQLite/JSONL)
- Wraps a FAISS index
- Metadata: Prefers SQLite (Disk) -> Falls back to JSONL (RAM)
- Integrates GraphService for Hybrid RAG (Structured + Unstructured)
- Uses EmbedCacheService to avoid re-embedding identical texts
"""

from __future__ import annotations
import json
import os
import logging
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss

# absolute imports
from backend.services.embed_cache_service import EmbedCacheService
from backend.core.config import settings
from backend.tools.embedder import Embedder

# Import the GraphService
try:
    from backend.services.graph_service import GraphService
except ImportError:
    GraphService = None

logger = logging.getLogger("agentic-rag.retrieve")

# Default paths
DEFAULT_INDEX = settings.FAISS_INDEX_PATH if hasattr(settings, "FAISS_INDEX_PATH") else "backend/db/vector_data/knowledge_faiss.index"
DEFAULT_META_JSONL = settings.FAISS_META_PATH if hasattr(settings, "FAISS_META_PATH") else "backend/db/vector_data/knowledge_meta.jsonl"
DEFAULT_DB_PATH = settings.META_DB_PATH if hasattr(settings, "META_DB_PATH") else "backend/db/vector_data/metadata_store.db" 
DEFAULT_EMBED_MODEL = settings.EMBEDDING_MODEL if hasattr(settings, "EMBEDDING_MODEL") else "mxbai-embed-large:latest"


def _load_meta_lines(meta_path: str) -> List[Dict[str, Any]]:
    """Legacy helper to load JSONL into memory."""
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
                meta.append({"text": ln})
    return meta


class RetrieveService:
    def __init__(
        self,
        index_path: str = DEFAULT_INDEX,
        meta_path: str = DEFAULT_META_JSONL,   
        db_path: str = DEFAULT_DB_PATH,       
        embed_cache: Optional[EmbedCacheService] = None,
        embedder: Optional[Embedder] = None,
        reranker_obj: Optional[Any] = None,
        reranker_enabled: bool = False,
        graph_service: Optional[GraphService] = None,
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.db_path = db_path
        self.reranker = reranker_obj
        self.reranker_enabled = bool(reranker_enabled)

        # -------------------------------------------------------
        # 1. Setup Metadata Store (SQLite preferred, JSONL fallback)
        # -------------------------------------------------------
        self.db_conn = None
        self.meta: List[Dict[str, Any]] = []

        if os.path.exists(self.db_path):
            # Option A: SQLite (Disk-based, low RAM)
            try:
                self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.db_conn.row_factory = sqlite3.Row
                logger.info(f"RetrieveService connected to SQLite: {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to connect to SQLite DB: {e}")
        
        # If no DB connection, fall back to loading JSONL into memory
        if self.db_conn is None:
            logger.info(f"SQLite DB not found/active. Falling back to JSONL: {self.meta_path}")
            self.meta = _load_meta_lines(self.meta_path)
            logger.info("Loaded metadata entries (RAM): %d", len(self.meta))

        # -------------------------------------------------------
        # 2. Load FAISS index lazily
        # -------------------------------------------------------
        self._index: Optional[faiss.Index] = None
        self._index_ntotal: int = 0

        # 3. Embedder / Cache
        self.embedder = embedder or Embedder()
        self.embed_model = getattr(self.embedder, "model", DEFAULT_EMBED_MODEL)
        self.embed_cache = embed_cache or EmbedCacheService()

        # 4. Check FAISS index existence
        self._ensure_index()

        # -------------------------------------------------------
        # 5. Initialize Graph Integration
        # -------------------------------------------------------
        self.graph_service = graph_service
        self.neo4j_vector = None
        
        if self.graph_service:
            try:
                self.neo4j_vector = self.graph_service.get_vector_index()
                logger.info("RetrieveService connected to Neo4j Vector Store.")
            except Exception as e:
                logger.error(f"Failed to attach Neo4j Vector Store: {e}")

    # --------------------------
    # Helper: Unified Metadata Access
    # --------------------------
    def _get_meta_by_id(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves metadata for a chunk ID from either SQLite or the in-memory list.
        """
        if idx < 0:
            return {}

        # 1. Try SQLite
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                # Adjust columns based on your schema
                cursor.execute(
                    "SELECT chunk_id, doc_name, text, start_token, end_token, pid, block_id FROM chunks WHERE chunk_id = ?", 
                    (idx,)
                )
                row = cursor.fetchone()
                return dict(row) if row else {}
            except Exception as e:
                logger.error(f"Error fetching chunk_id={idx} from DB: {e}")
                return {}

        # 2. Try In-Memory List
        if 0 <= idx < len(self.meta):
            return self.meta[idx]

        return {}

    # --------------------------
    # Hybrid Retrieval (Graph + Vector)
    # --------------------------
    def retrieve_hybrid(self, query: str, top_k: int = settings.TOP_K_RETRIEVAL) -> List[str]:
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
            used_source = "FAISS (SQLite)" if self.db_conn else "FAISS (RAM)"
            
            # Embed and search
            qvec = self.embed_text(query).astype("float32")
            D, I = self.search_vector(qvec, fetch_k)
            
            if I.size > 0:
                for idx in I[0]:
                    if idx < 0: continue
                    # Unified Metadata Fetch
                    meta = self._get_meta_by_id(int(idx))
                    text = meta.get('text', '')
                    if text:
                        vector_candidates.append(text)

        # 3. RERANKING STEP
        if self.reranker_enabled and self.reranker and vector_candidates:
            try:
                print(f"--- [RERANKER] Scoring {len(vector_candidates)} docs from {used_source} ---")
                # scores = self.reranker.score(query, vector_candidates)
                # Note: Assuming your reranker uses .score() or similar wrapper
                # If using CrossEncoder directly: model.predict([[query, doc] for doc in docs])
                scores = self.reranker.score(query, vector_candidates)
                
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
                docs.extend(vector_candidates[:top_k])
        else:
            docs.extend(vector_candidates[:top_k])

        return docs

    # --------------------------
    # Standard Retrieval (Legacy/API)
    # --------------------------
    def _build_result_record(self, idx: int, score: float) -> Dict[str, Any]:
        """Builds result dict using unified metadata fetcher."""
        meta = self._get_meta_by_id(idx)
        return {"index": idx, "score": float(score), "meta": meta}

    def retrieve(self, query: str, top_k: int = settings.TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Standard retrieval returning dicts with metadata.
        """
        # embed query
        qvec = self.embed_text(query).astype("float32")

        # if reranker enabled
        if self.reranker_enabled and self.reranker:
            initial_k = max(getattr(settings, "RERANKER_INITIAL_K", 20), top_k)
            D, I = self.search_vector(qvec, initial_k)
            candidates = []
            if I.size > 0:
                for dist, idx in zip(D[0], I[0]):
                    if idx < 0: continue
                    candidates.append(self._build_result_record(int(idx), float(dist)))
            
            try:
                print(f"--- [RERANKER] Scoring {len(candidates)} documents for query: '{query}' ---")
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
                    if idx < 0: continue
                    results.append(self._build_result_record(int(idx), float(dist)))
            return results

    def retrieve_batch(self, queries: List[str], top_k: int = settings.TOP_K_RETRIEVAL) -> List[List[Dict[str, Any]]]:
        if not queries:
            return [[] for _ in queries]

        vecs = self.embed_batch(queries)
        out = []
        for v in vecs:
            D, I = self.search_vector(v, top_k)
            res = []
            if I.size > 0:
                for dist, idx in zip(D[0], I[0]):
                    if idx < 0: continue
                    res.append(self._build_result_record(int(idx), float(dist)))
            out.append(res)
        return out

    # --------------------------
    # Embedding helpers (with cache)
    # --------------------------
    def embed_text(self, text: str) -> np.ndarray:
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
                    logger.info("FAISS index loaded. ntotal=%d", self._index_ntotal)
                except Exception as e:
                    logger.exception("Failed to load FAISS index lazily: %s", e)
                    self._index = None
            else:
                logger.warning("FAISS index not found at %s", self.index_path)

    def search_vector(self, vec: np.ndarray, top_k: int = settings.TOP_K_RETRIEVAL) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_index()
        if self._index is None:
            return np.array([[]]), np.array([[]])
            
        q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        D, I = self._index.search(q, top_k)
        return D, I

    # --------------------------
    # Utilities
    # --------------------------
    def index_count(self) -> int:
        return self._index_ntotal

    def close(self):
        # Close SQLite
        if hasattr(self, "db_conn") and self.db_conn:
            try:
                self.db_conn.close()
                logger.info("Closed SQLite metadata connection")
            except Exception:
                pass

        # Close Cache
        try:
            self.embed_cache.close()
        except Exception:
            pass