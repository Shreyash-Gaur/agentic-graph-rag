"""
Robust configuration for Agentic Graph-RAG.

Allows extra env variables (ignored), parses CORS flexibly,
and preserves correct defaults matching the project's .env values.
"""

from __future__ import annotations
from pydantic_settings import BaseSettings
from typing import List, Optional
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
    # --- 1. API ---
    API_TITLE: str = "Agentic Graph-RAG API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    CORS_ORIGINS: Optional[str] = None

    # --- 2. Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    EMBEDDING_MODEL: str = "mxbai-embed-large:latest"

    # --- 3. RAG & Generation ---
    MAX_TOKENS: int = 1024
    MAX_ITERATIONS: int = 7
    TOP_K_RETRIEVAL: int = 5

    # --- 4. Vector chunking (ingest_vector_watch.py) ---
    CHUNK_TOKENS: int = 512
    CHUNK_OVERLAP: int = 100
    EMBEDDING_BATCH_SIZE: int = 16

    # --- 5. Graph ingestion — semantic chunking (ingest_graph_watch.py) ---
    # SemanticChunker splits on meaning boundaries using embedding similarity.
    # GRAPH_CHUNK_TOKENS and GRAPH_CHUNK_OVERLAP are removed — SemanticChunker
    # ignores fixed token counts. Chunk sizes vary by semantic content.
    SEMANTIC_CHUNK_THRESHOLD_TYPE: str = "percentile"
    # 85 = split when sentence similarity drops below the 85th percentile.
    # Lower = more splits (smaller chunks). Higher = fewer splits (larger chunks).
    SEMANTIC_CHUNK_BREAKPOINT: int = 85

    # --- 6. Graph ingestion — coreference resolution ---
    # One extra LLM call per chunk. Resolves pronouns and partial names to full
    # canonical forms before graph extraction runs.
    # Set false for faster ingestion during development.
    USE_COREF: bool = True

    # --- 7. Neo4j ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # --- 8. Watcher ---
    WATCH_DIR: str = "knowledge"

    # --- 9. File paths ---
    FAISS_INDEX_PATH: str = "backend/db/vector_data/knowledge_faiss.index"
    FAISS_META_PATH: str = "backend/db/vector_data/knowledge_meta.jsonl"
    META_DB_PATH: str = "backend/db/vector_data/metadata_store.db"
    MEMORY_DB_PATH: str = "backend/db/memory/memory_store.sqlite"
    EMBEDDING_CACHE_DB: str = "backend/db/embedding_cache/embed_cache.sqlite"
    MEMORY_MAX_TURNS: int = 20

    # --- 10. Reranker ---
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_INITIAL_K: int = 15
    RERANKER_BACKEND: str = "cross-encoder"
    RERANKER_NORMALIZE: str = "sigmoid"
    RERANKER_BATCH_SIZE: int = 8

    # --- 11. Semantic cache ---
    SEMANTIC_CACHE_MODEL: str = "BAAI/bge-large-en-v1.5"
    SEMANTIC_CACHE_THRESHOLD: float = 0.85

    # --- 12. Feature flags ---
    USE_HYDE: bool = True
    CHAINLIT_ENABLED: bool = True

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


settings = Settings()