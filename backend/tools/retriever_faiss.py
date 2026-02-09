# backend/tools/retriever_faiss.py
from pathlib import Path
import json
import numpy as np
import faiss
import os
from typing import List, Dict, Any, Optional

# Attempt to import the repo embedder (preferred) or fall back to sentence-transformers
_EMBEDDER = None
_EMBEDDER_NAME = None
try:
    from .embedder import Embedder as _RepoEmbedder
    _repo = _RepoEmbedder()
    _EMBEDDER_NAME = "repo_embedder"
    def _embed_texts(texts: List[str]) -> np.ndarray:
        # prefer batch API if available
        if hasattr(_repo, "embed_batch"):
            arr = _repo.embed_batch(texts)
        elif hasattr(_repo, "embed"):
            arr = [_repo.embed(t) for t in texts]
        elif callable(_repo):
            arr = _repo(texts)
        else:
            raise RuntimeError("Repo embedder found but no usable API")
        import numpy as _np
        return _np.asarray([_np.asarray(v, dtype=_np.float32) for v in arr], dtype=_np.float32)
except Exception:
    _EMBEDDER_NAME = "sentence-transformers/fallback"
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-mpnet-base-v2")
        def _embed_texts(texts: List[str]) -> np.ndarray:
            arr = _st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return arr.astype("float32")
    except Exception:
        def _embed_texts(texts: List[str]) -> np.ndarray:
            raise RuntimeError("No embedder available (install sentence-transformers or fix backend/tools/embedder.py)")

class FAISSRetriever:
    def __init__(self, index_path: str, meta_path: str):
        """
        index_path: path to faiss index file (faiss.write_index)
        meta_path: path to jsonl file (one JSON per vector in same order)
        """
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        # Load index and metadata
        self.index: faiss.Index = faiss.read_index(str(self.index_path))
        self.meta: List[Dict[str, Any]] = [json.loads(l) for l in open(self.meta_path, "r", encoding="utf8")]
        if len(self.meta) != self.index.ntotal:
            print(f"Warning: metadata count ({len(self.meta)}) != index.ntotal ({self.index.ntotal})")
        print(f"FAISS retriever initialized. index.ntotal = {self.index.ntotal}, embedder = {_EMBEDDER_NAME}")

    def reload(self):
        """Reload index & metadata from disk (useful if index updated externally)."""
        self.index = faiss.read_index(str(self.index_path))
        self.meta = [json.loads(l) for l in open(self.meta_path, "r", encoding="utf8")]

    def embed_query(self, query: str) -> np.ndarray:
        vec = _embed_texts([query])
        return vec.astype("float32")

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Batch embedding of queries. Returns np.ndarray shape (len(queries), dim)."""
        vecs = _embed_texts(queries)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k results for single query.
        Returns list of dicts: [{ 'score': float, 'index': int, 'meta': {...} }, ...]
        """
        qvec = self.embed_query(query)
        D, I = self.index.search(qvec, k=top_k)
        results = []
        for dist, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            meta = self.meta[idx] if idx < len(self.meta) else {}
            results.append({
                "score": float(dist),
                "index": int(idx),
                "meta": meta
            })
        return results

    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieve: embed all queries and perform one faiss.search call.
        Returns a list (len = len(queries)) of lists of result dicts.
        """
        if not queries:
            return []
        qvecs = self.embed_queries(queries)  # shape (nq, dim)
        D, I = self.index.search(qvecs, k=top_k)
        batch_results: List[List[Dict[str, Any]]] = []
        for rowD, rowI in zip(D, I):
            one = []
            for dist, idx in zip(rowD.tolist(), rowI.tolist()):
                if idx < 0:
                    continue
                meta = self.meta[idx] if idx < len(self.meta) else {}
                one.append({"score": float(dist), "index": int(idx), "meta": meta})
            batch_results.append(one)
        return batch_results

    def retrieve_batch_with_scores(self, queries: List[str], top_k: int = 5):
        """Alias for retrieve_batch (keeps naming explicit)."""
        return self.retrieve_batch(queries, top_k=top_k)
