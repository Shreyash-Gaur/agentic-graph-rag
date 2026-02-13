
# backend/models/request_models.py

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from backend.core.config import settings


# ------------------------------
# Basic RAG Query
# ------------------------------
class QueryRequest(BaseModel):
    """
    Request schema for /query and /agent_query endpoints.
    """
    query: str
    top_k: int = Field(default=settings.TOP_K_RETRIEVAL, ge=1, le=50)
    system_prompt: Optional[str] = None
    conversation_id: Optional[str] = "default"
    max_tokens: int = Field(default=settings.MAX_TOKENS, ge=1, le=4096)
    temperature: float = 0.0


# ------------------------------
# Retrieval-only request
# ------------------------------
class RetrieveRequest(BaseModel):
    """
    Request schema for /retrieve endpoint.
    """
    query: str
    top_k: int = Field(default=settings.TOP_K_RETRIEVAL, ge=1, le=50)


# ------------------------------
# Iterative Agentic Query (multi-step)
# ------------------------------
class IterativeQueryRequest(BaseModel):
    """
    Request schema for agent-based iterative reasoning.
    """
    query: str
    conversation_id: Optional[str] = "default"
    top_k: int = Field(default=settings.TOP_K_RETRIEVAL, ge=1, le=50)
    max_iterations: int = Field(default=settings.MAX_ITERATIONS, ge=1, le=10)
    temperature: float = 0.0


# ------------------------------
# Multi-document ingestion
# ------------------------------
class IngestRequest(BaseModel):
    """
    Request schema for ingesting a single PDF/doc.
    """
    file_path: str
    chunk_tokens: int = Field(default=settings.CHUNK_TOKENS, ge=1, le=1024)
    overlap: int = Field(default=settings.CHUNK_OVERLAP, ge=1, le=1024)


class BatchIngestRequest(BaseModel):
    """
    Request schema for ingesting multiple documents.
    """
    file_paths: List[str]
    chunk_tokens: int = Field(default=settings.CHUNK_TOKENS, ge=1, le=1024)
    overlap: int = Field(default=settings.CHUNK_OVERLAP, ge=1, le=1024)


# ------------------------------
# Embedding Cache Management
# ------------------------------
class CacheLookupRequest(BaseModel):
    """
    For testing if a text is already embedded.
    """
    text: str
