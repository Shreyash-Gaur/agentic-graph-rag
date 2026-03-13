# backend/main.py

from __future__ import annotations

import logging
import os
import sys
import time
import shlex
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config     import settings
from backend.core.logger     import setup_logging
from backend.core.exceptions import AgenticRAGException

from backend.services.retrieve_service       import RetrieveService
from backend.services.embed_cache_service    import EmbedCacheService
from backend.services.memory_service         import MemoryService
from backend.services.semantic_cache_service import SemanticCacheService

try:
    from backend.services.graph_service import GraphService
except Exception as e:
    print(f"GraphService import failed: {e}")
    GraphService = None

from backend.agents.graph_agent import GraphRAGAgent

from backend.models.request_models  import QueryRequest, RetrieveRequest
from backend.models.response_models import QueryResponse, RetrieveResponse, DocumentResult

setup_logging()
logger = logging.getLogger("agentic-rag.api")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

retrieve_service: Optional[RetrieveService]     = None
embed_cache:      Optional[EmbedCacheService]   = None
memory_service:   Optional[MemoryService]       = None
semantic_cache:   Optional[SemanticCacheService] = None
graph_service:    Optional[GraphService]        = None
rag_agent:        Optional[GraphRAGAgent]       = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_service, semantic_cache, embed_cache, retrieve_service, graph_service, rag_agent

    logger.info("Starting Agentic-RAG service...")

    try:
        memory_service = MemoryService(
            max_history=settings.MEMORY_MAX_TURNS,
            use_sqlite=True,
            db_path=settings.MEMORY_DB_PATH,
            preload=False,
        )
        logger.info("MemoryService ready.")
    except Exception as e:
        logger.exception("MemoryService init failed: %s", e)

    try:
        semantic_cache = SemanticCacheService()
        logger.info("SemanticCacheService ready.")
    except Exception as e:
        logger.exception("SemanticCacheService init failed: %s", e)

    try:
        embed_cache = EmbedCacheService(db_path=settings.EMBEDDING_CACHE_DB)
        logger.info("EmbedCacheService ready.")
    except Exception as e:
        logger.exception("EmbedCacheService init failed: %s", e)

    reranker_obj = None
    if settings.RERANKER_ENABLED:
        try:
            from backend.tools.reranker import Reranker
            reranker_obj = Reranker()
            logger.info("Reranker loaded: %s", settings.RERANKER_MODEL)
        except Exception as e:
            logger.exception("Reranker init failed: %s", e)

    if GraphService:
        try:
            graph_service = GraphService()
            logger.info("GraphService connected to Neo4j.")
        except Exception as e:
            logger.error("GraphService init failed (Neo4j down?): %s", e)
            graph_service = None
    else:
        logger.warning("GraphService not available.")

    try:
        retrieve_service = RetrieveService(
            embed_cache=embed_cache,
            embedder=None,
            reranker_obj=reranker_obj,
            reranker_enabled=bool(reranker_obj),
            graph_service=graph_service,
        )
        logger.info("RetrieveService ready (index=%s).", settings.FAISS_INDEX_PATH)
    except Exception as e:
        logger.exception("RetrieveService init failed: %s", e)

    if retrieve_service:
        try:
            rag_agent = GraphRAGAgent(
                retrieve_service=retrieve_service,
                model_name=settings.OLLAMA_MODEL,
            )
            logger.info("GraphRAGAgent ready.")
        except Exception as e:
            logger.exception("GraphRAGAgent init failed: %s", e)
    else:
        logger.warning("RetrieveService missing — GraphRAGAgent not started.")

    yield

    logger.info("Shutting down...")
    for svc in [retrieve_service, embed_cache, memory_service, graph_service]:
        if svc and hasattr(svc, "close"):
            try:
                svc.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

APP = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION, lifespan=lifespan)

# FIX: allow_credentials=True with allow_origins=["*"] is rejected by browsers
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@APP.get("/health")
def health():
    return {
        "status":        "ok",
        "retriever":     bool(retrieve_service),
        "rag_agent":     bool(rag_agent),
        "graph_service": bool(graph_service),
        "semantic_cache": bool(semantic_cache),
    }


@APP.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    if not retrieve_service:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    try:
        doc_strings = retrieve_service.retrieve_hybrid(req.query, top_k=req.top_k)
        doc_results = [
            DocumentResult(text=text, score=1.0, metadata={}, source="graph/vector", chunk_id=i)
            for i, text in enumerate(doc_strings)
        ]
        return RetrieveResponse(query=req.query, results=doc_results, num_results=len(doc_results))
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(e))


@APP.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")

    try:
        session_id = req.conversation_id or "default"
        user_query = req.query

        # 1. Semantic cache — skip when any of these are true:
        #    - mode=detailed (long answer button)
        #    - temperature > 0.1 (creative answer button)
        #    - max_tokens > default (long answer sends 2x)
        #    - bypass_cache=True (explicit override)
        use_cache = (
            semantic_cache
            and req.mode == "concise"
            and req.temperature <= 0.1
            and req.max_tokens <= settings.MAX_TOKENS
            and not req.bypass_cache
        )
        if use_cache:
            cached = semantic_cache.check_cache(user_query)
            if cached:
                logger.info("Serving from semantic cache.")
                if memory_service:
                    memory_service.add_turn(session_id, user_query, cached)
                return QueryResponse(
                    query=user_query,
                    answer=cached,
                    sources=[],
                    num_sources=0,
                    prompt="semantic_cache",
                    metadata={"cached": True},
                )

        # 2. Memory context
        chat_history = ""
        if memory_service:
            chat_history = memory_service.get_context(session_id, last_n=10)

        # 3. Run agent — mode comes from request field, not inferred from max_tokens
        output     = rag_agent.query(
            query=user_query,
            mode=req.mode,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            chat_history=chat_history,
        )
        ai_answer   = output.get("answer", "No answer generated.")
        raw_sources = output.get("sources", [])

        # 4. Persist
        if memory_service:
            try:
                memory_service.add_turn(session_id, user_query, ai_answer)
                if semantic_cache and req.mode == "concise" and not req.bypass_cache:
                    semantic_cache.add_new_turn(user_query, ai_answer)
            except Exception as e:
                logger.error("Failed to save to memory: %s", e)

        # 5. Build source list — FIX: was always sources=[]
        doc_results = []
        for i, text in enumerate(raw_sources):
            if not text:
                continue
            doc_results.append(DocumentResult(
                text=str(text)[:500],
                score=1.0,
                metadata={},
                source="graph/vector",
                chunk_id=i,
            ))

        return QueryResponse(
            query=user_query,
            answer=ai_answer,
            sources=doc_results,
            num_sources=len(doc_results),
            prompt="",
            metadata=output.get("metadata", {}),
        )

    except Exception as e:
        logger.exception("Agent query failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

INGEST_UPLOAD_DIR = Path(settings.WATCH_DIR)
INGEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@APP.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    filename = Path(file.filename).name
    out_path = INGEST_UPLOAD_DIR / filename
    try:
        with out_path.open("wb") as fh:
            fh.write(await file.read())
        return {
            "status":   "accepted",
            "filename": filename,
            "note":     "File saved. Watcher will detect and ingest it shortly.",
        }
    except Exception as e:
        logger.error("Failed to save upload: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:APP", host="0.0.0.0", port=8000, reload=settings.DEBUG)