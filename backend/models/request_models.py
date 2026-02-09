
# backend/models/request_models.py

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ------------------------------
# Basic RAG Query
# ------------------------------
class QueryRequest(BaseModel):
    """
    Request schema for /query and /agent_query endpoints.
    """
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    system_prompt: Optional[str] = None
    conversation_id: Optional[str] = "default"
    max_tokens: int = 512
    temperature: float = 0.0


# ------------------------------
# Retrieval-only request
# ------------------------------
class RetrieveRequest(BaseModel):
    """
    Request schema for /retrieve endpoint.
    """
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


# ------------------------------
# Iterative Agentic Query (multi-step)
# ------------------------------
class IterativeQueryRequest(BaseModel):
    """
    Request schema for agent-based iterative reasoning.
    """
    query: str
    conversation_id: Optional[str] = "default"
    top_k: int = 5
    max_iterations: int = 3
    temperature: float = 0.0


# ------------------------------
# Multi-document ingestion
# ------------------------------
class IngestRequest(BaseModel):
    """
    Request schema for ingesting a single PDF/doc.
    """
    file_path: str
    chunk_tokens: int = 512
    overlap: int = 128


class BatchIngestRequest(BaseModel):
    """
    Request schema for ingesting multiple documents.
    """
    file_paths: List[str]
    chunk_tokens: int = 512
    overlap: int = 128


# ------------------------------
# Embedding Cache Management
# ------------------------------
class CacheLookupRequest(BaseModel):
    """
    For testing if a text is already embedded.
    """
    text: str
