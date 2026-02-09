# backend/core/config.py
"""
Robust configuration for Agentic-RAG.

This version allows extra env variables (ignored), parses CORS flexibly,
and preserves your chosen defaults for OLLAMA_MODEL and EMBEDDING_MODEL.
"""

from __future__ import annotations
from pydantic_settings import BaseSettings
from typing import List, Optional, Any
import os, json

def _parse_cors(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    if raw == "":
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
    except Exception:
        pass
    return [p.strip() for p in raw.split(",") if p.strip()]

class Settings(BaseSettings):
    # Basic
    API_TITLE: str = "Agentic RAG API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Raw string for CORS; use settings.CORS property to get parsed list
    CORS_ORIGINS: Optional[str] = None

    # Ollama / Embedding (your chosen defaults)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # RAG Behavior
    TOP_K_RETRIEVAL: int = 5
    CHUNK_TOKENS: int = 512
    CHUNK_OVERLAP: int = 128

    # Neo4j Config
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Paths
    FAISS_INDEX_PATH: str = "backend/db/knowledge_faiss.index"
    FAISS_META_PATH: str = "backend/db/knowledge_meta.jsonl"

    # Memory & cache
    MEMORY_DB_PATH: str = "backend/db/memory_store.sqlite"
    EMBEDDING_CACHE_DB: str = "backend/db/embed_cache.sqlite"
    MEMORY_MAX_TURNS: int = 20

    # Reranker
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_INITIAL_K: int = 20

    # Chainlit
    CHAINLIT_ENABLED: bool = True

    # Pydantic v2 configuration: ignore extra env keys and declare env file
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }

    @property
    def CORS(self) -> List[str]:
        raw = os.getenv("CORS_ORIGINS", None)
        if raw is None and self.CORS_ORIGINS is not None:
            raw = self.CORS_ORIGINS
        return _parse_cors(raw)


# global instance
settings = Settings()
