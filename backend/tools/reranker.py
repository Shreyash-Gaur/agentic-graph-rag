# backend/tools/reranker.py
import os
import numpy as np
from typing import List, Dict
from sentence_transformers import CrossEncoder
import torch


class CrossEncoderReranker:
    """
    GPU-accelerated Cross-Encoder Reranker.
    Uses MS MARCO MiniLM cross-encoder (accurate + fast).
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # auto-select GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[RERANKER] Loading {self.model_name} on device={self.device}")
        self.model = CrossEncoder(self.model_name, device=self.device)

    def score(self, query: str, docs: List[str], batch_size: int = 16) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)

        pairs = [(query, d) for d in docs]
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )
        return np.asarray(scores, dtype=np.float32)

    def rerank(self, query: str, results: List[Dict], top_k: int):
        texts = [r["meta"].get("text", "") for r in results]
        scores = self.score(query, texts)

        for r, s in zip(results, scores):
            r["_rerank_score"] = float(s)

        results_sorted = sorted(results, key=lambda x: x["_rerank_score"], reverse=True)
        return results_sorted[:top_k]
