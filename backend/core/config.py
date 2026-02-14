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
    OLLAMA_MODEL: str = "phi3:mini"
    EMBEDDING_MODEL: str = "mxbai-embed-large:latest"

# Vector / General RAG Behavior
    MAX_TOKENS: int = 512
    MAX_ITERATIONS: int = 6
    TOP_K_RETRIEVAL: int = 4
    
    # Vector Chunking
    CHUNK_TOKENS: int = 512
    CHUNK_OVERLAP: int = 128
    EMBEDDING_BATCH_SIZE: int = 32

    # NEW: Graph Chunking
    GRAPH_CHUNK_TOKENS: int = 256
    GRAPH_CHUNK_OVERLAP: int = 32

    # Neo4j Config
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Watcher folder
    WATCH_DIR: str = "knowledge"

    # Paths
    FAISS_INDEX_PATH: str = "backend/db/vector_data/knowledge_faiss.index"
    FAISS_META_PATH: str = "backend/db/vector_data/knowledge_meta.jsonl"
    META_DB_PATH: str = "backend/db/vector_data/metadata_store.db"

    # Memory & cache
    MEMORY_DB_PATH: str = "backend/db/memory/memory_store.sqlite"
    EMBEDDING_CACHE_DB: str = "backend/db/embedding_cache/embed_cache.sqlite"
    MEMORY_MAX_TURNS: int = 100

    # Semantic Cache
    SEMANTIC_CACHE_MODEL: str = "BAAI/bge-large-en-v1.5"
    SEMANTIC_CACHE_THRESHOLD: float = 0.80

    # Reranker
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_INITIAL_K: int = 14

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
