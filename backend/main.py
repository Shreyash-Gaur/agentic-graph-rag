import os
import sys
import time
import shlex
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# --- Core Imports ---
from backend.core.config import settings
from backend.core.logger import setup_logging
from backend.core.exceptions import AgenticRAGException

# --- Services ---
from backend.services.retrieve_service import RetrieveService
from backend.services.embed_cache_service import EmbedCacheService
from backend.services.memory_service import MemoryService
from backend.tools.ollama_client import OllamaClient

# --- Graph Service Import ---
try:
    from backend.services.graph_service import GraphService
except Exception as e:
    print(f"CRITICAL ERROR IMPORTING GRAPH SERVICE: {e}")
    GraphService = None

# --- Agents ---
from backend.agents.graph_agent import GraphRAGAgent

# --- Models ---
from backend.models.request_models import QueryRequest, RetrieveRequest
from backend.models.response_models import QueryResponse, RetrieveResponse, DocumentResult

# ------------------------------
# Setup
# ------------------------------
setup_logging()
logger = logging.getLogger("agentic-rag")

APP = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# CORS
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Service Instances
retrieve_service: Optional[RetrieveService] = None
embed_cache: Optional[EmbedCacheService] = None
memory_service: Optional[MemoryService] = None
ollama_client: Optional[OllamaClient] = None
graph_service: Optional[GraphService] = None  # <--- NEW Global

# Global Agent Instance
rag_agent: Optional[GraphRAGAgent] = None


@APP.on_event("startup")
def startup_event():
    """
    Initialize services and the Graph Agent.
    """
    global memory_service, embed_cache, retrieve_service, ollama_client, graph_service
    global rag_agent

    logger.info("Starting Agentic-RAG service...")

    # 1. Memory Service
    try:
        memory_service = MemoryService(
            max_history=settings.MEMORY_MAX_TURNS,
            use_sqlite=True,
            db_path=settings.MEMORY_DB_PATH,
            preload=False,
        )
        logger.info(f"MemoryService initialized: {settings.MEMORY_DB_PATH}")
    except Exception as e:
        logger.exception(f"Failed to init MemoryService: {e}")

    # 2. Embed Cache
    try:
        embed_cache = EmbedCacheService(db_path=settings.EMBEDDING_CACHE_DB)
        logger.info(f"EmbedCacheService initialized: {settings.EMBEDDING_CACHE_DB}")
    except Exception as e:
        logger.exception(f"Failed to init EmbedCacheService: {e}")

    # 3. Reranker (Lazy loaded inside RetrieveService if passed)
    reranker_obj = None
    if settings.RERANKER_ENABLED:
        try:
            from backend.tools.reranker import CrossEncoderReranker
            reranker_obj = CrossEncoderReranker(model_name=settings.RERANKER_MODEL)
            logger.info(f"Reranker loaded: {settings.RERANKER_MODEL}")
        except Exception as e:
            logger.exception(f"Failed to load reranker: {e}")

    # 4. Ollama Client
    try:
        ollama_client = OllamaClient(base_url=settings.OLLAMA_BASE_URL)
        logger.info(f"Ollama client ready at {settings.OLLAMA_BASE_URL}")
    except Exception as e:
        logger.exception(f"Failed to init OllamaClient: {e}")

    # 5. Graph Service (NEW)
    if GraphService:
        try:
            graph_service = GraphService()
            logger.info("GraphService initialized and connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to init GraphService (Neo4j might be down): {e}")
            graph_service = None
    else:
        logger.warning("GraphService class not found. Skipping Neo4j init.")

    # 6. Retrieve Service (Updated to accept graph_service)
    try:
        retrieve_service = RetrieveService(
            index_path=settings.FAISS_INDEX_PATH,
            meta_path=settings.FAISS_META_PATH,
            embed_cache=embed_cache,
            embedder=None,
            reranker_obj=reranker_obj,
            reranker_enabled=bool(reranker_obj),
            graph_service=graph_service  # <--- INJECTED HERE
        )
        logger.info(f"RetrieveService initialized (index={settings.FAISS_INDEX_PATH})")
    except Exception as e:
        logger.exception(f"Failed to init RetrieveService: {e}")

    # 7. Initialize Graph Agent
    if retrieve_service:
        try:
            rag_agent = GraphRAGAgent(
                retrieve_service=retrieve_service,
                model_name=settings.OLLAMA_MODEL
            )
            logger.info("GraphRAGAgent successfully initialized.")
        except Exception as e:
            logger.exception(f"Failed to init GraphRAGAgent: {e}")
    else:
        logger.warning("RetrieveService missing. GraphRAGAgent not initialized.")


@APP.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down Agentic-RAG service...")
    for svc in [retrieve_service, embed_cache, memory_service, graph_service]:
        if svc and hasattr(svc, "close"):
            try:
                svc.close()
            except Exception:
                pass


@APP.get("/health")
def health():
    return {
        "status": "ok",
        "retriever": bool(retrieve_service),
        "rag_agent": bool(rag_agent),
        "ollama": bool(ollama_client),
        "graph_service": bool(graph_service)
    }


# ------------------------------
# Endpoints
# ------------------------------

@APP.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    """
    Direct retrieval endpoint. 
    NOTE: Using 'retrieve_hybrid' here to show graph results in direct tests.
    """
    if not retrieve_service:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        # Use Hybrid Retrieve to get strings directly
        doc_strings = retrieve_service.retrieve_hybrid(req.query, top_k=req.top_k)
        
        # Convert to Pydantic DocumentResult for response consistency
        # Since hybrid returns strings, we mock the metadata/score for this view
        doc_results = []
        for i, text in enumerate(doc_strings):
            doc = DocumentResult(
                text=text,
                score=1.0, # Placeholder
                metadata={},
                source="graph/vector",
                chunk_id=i
            )
            doc_results.append(doc)

        return RetrieveResponse(
            query=req.query,
            results=doc_results,
            num_results=len(doc_results)
        )
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(e))

@APP.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Main RAG endpoint.
    """
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")

    try:
        # Determine mode
        mode = "detailed" if req.max_tokens > 512 else "concise"
        
        output = rag_agent.query(
            query=req.query, 
            mode=mode, 
            temperature=req.temperature
        )
        answer_text = output.get("answer", "No answer generated.")

        if memory_service:
            session_id = getattr(req, "session_id", "default_session")
            
            try:
                memory_service.add_turn(
                    session_id=session_id,
                    user_input=req.query,
                    ai_output=answer_text
                )
                logger.info(f"Saved turn to memory for session: {session_id}")
            except Exception as e:
                logger.error(f"Failed to save to memory: {e}")

        return QueryResponse(
            query=req.query,
            answer=output.get("answer", "No answer generated."),
            sources=[], 
            num_sources=0,
            prompt="",
            metadata=output.get("metadata", {})
        )

    except Exception as e:
        logger.exception(f"Agent Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Ingestion Endpoints
# ------------------------------
INGEST_UPLOAD_DIR = Path("knowledge")
INGEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _run_ingest_subprocess(cmd_list: List[str], env: Optional[Dict[str, str]] = None, log_prefix: str = "ingest") -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    
    log_dir = repo_root / "logs" / "ingests"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd_str = " ".join(shlex.quote(p) for p in cmd_list)
    ts = int(time.time())
    out_log = log_dir / f"{log_prefix}-{ts}.out.log"
    err_log = log_dir / f"{log_prefix}-{ts}.err.log"

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    # Ensure pythonpath includes the root
    if "PYTHONPATH" not in full_env:
        full_env["PYTHONPATH"] = str(repo_root)

    out_f = open(out_log, "ab")
    err_f = open(err_log, "ab")

    p = subprocess.Popen(cmd_list, env=full_env, stdout=out_f, stderr=err_f, start_new_session=True)

    return {"pid": p.pid, "out_log": str(out_log), "err_log": str(err_log), "cmd": cmd_str}

@APP.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Uploads a file and triggers the GRAPH ingestion script.
    """
    filename = Path(file.filename).name
    out_path = INGEST_UPLOAD_DIR / filename
    with out_path.open("wb") as fh:
        content = await file.read()
        fh.write(content)

    # Trigger Graph Ingestion Script
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "backend" / "scripts" / "ingest_graph.py"
    
    # Verify script exists, else fall back or error
    if not script_path.exists():
        # Fallback to old ingestion if graph script isn't there
        script_path = repo_root / "backend" / "scripts" / "ingest_multi_docs.py"

    cmd_list = [
        sys.executable,
        str(script_path),
        str(out_path.resolve()) # Argument for the file path
    ]
    
    env = {"PYTHONPATH": str(repo_root)}
    info = _run_ingest_subprocess(cmd_list, env=env, log_prefix="graph_ingest")

    return {
        "status": "accepted",
        "filename": filename,
        "pid": info["pid"],
        "logs": info["out_log"],
        "note": "Triggered graph ingestion"
    }