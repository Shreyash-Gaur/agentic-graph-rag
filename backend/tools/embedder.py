# backend/tools/embedder.py
import os
import requests
from requests.adapters import HTTPAdapter, Retry
from typing import List
import numpy as np
import json


class Embedder:
    """
    Ollama-backed embedder (supports only one text at a time).
    Uses /api/embeddings with the correct 'prompt' field.
    """

    def __init__(self, base_url=None, model=None, timeout=150):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
        self.timeout = timeout

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def _embed_one(self, text: str) -> np.ndarray:
        """Call Ollama embeddings endpoint for ONE text."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text  # IMPORTANT: must be 'prompt', NOT 'input'
        }

        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            # Possible shapes:
            # 1) {"embedding": [numbers]}
            if isinstance(data, dict) and "embedding" in data:
                return np.asarray(data["embedding"], dtype=np.float32)

            # 2) {"embeddings": [[...]]}
            if isinstance(data, dict) and "embeddings" in data:
                return np.asarray(data["embeddings"][0], dtype=np.float32)

            # Unknown shape → dump debug
            dbg = {
                "response": data,
                "note": "Unknown embedding response shape",
                "text": text
            }
            with open("backend/db/embedding_cache/embed_debug.json", "w") as f:
                json.dump(dbg, f, indent=2)
            raise RuntimeError(f"Unknown embedding response format. Saved to embed_debug.json.")

        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def embed(self, text: str) -> np.ndarray:
        """Single text embed."""
        return self._embed_one(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding = loop (Ollama does not support batches)."""
        vectors = [self._embed_one(t) for t in texts]

        # Ensure all embeddings have same dimension
        dim = max(len(v) for v in vectors)
        out = np.zeros((len(vectors), dim), dtype=np.float32)
        for i, v in enumerate(vectors):
            out[i, :len(v)] = v

        return out
