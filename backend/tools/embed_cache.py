# backend/tools/embed_cache.py
import os
import hashlib
import numpy as np
from pathlib import Path
import json

class EmbeddingCache:
    def __init__(self, cache_dir="backend/db/embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key_for_text(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return h

    def _path_for_key(self, key: str) -> Path:
        # store in subfolders to avoid too many files in one folder
        return self.cache_dir / (key[:2]) / f"{key}.npy"

    def get(self, key: str):
        p = self._path_for_key(key)
        if p.exists():
            try:
                arr = np.load(p)
                return arr.astype("float32")
            except Exception:
                return None
        return None

    def set(self, key: str, vector):
        p = self._path_for_key(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, np.asarray(vector, dtype="float32"))
