
# Agentic Graph-RAG: Hybrid Knowledge Retrieval & Multi-Step Reasoning

## Project Overview

Agentic Graph-RAG is a production-ready, highly optimized AI retrieval system that solves the fundamental limitations of standard vector-only RAG (hallucinations, context fragmentation, and multi-hop reasoning failures).

This system utilizes a **Hybrid Retrieval strategy**, fusing dense vector search (FAISS) for semantic similarity with structured knowledge graph traversal (Neo4j) for entity-relationship mapping. It is orchestrated by an autonomous LangGraph agent that evaluates document relevance, dynamically rewrites queries, and utilizes external tools (like mathematical calculators) to synthesize highly accurate responses. It is designed to act as an intelligent, context-aware backend for enterprise applications requiring precise domain knowledge extraction.

## Key Features

* **Agentic Orchestration (LangGraph):** Implements an advanced state machine with self-correction routing (Route  Retrieve  Grade  Transform  Generate).
* **Hybrid Retrieval Engine:** Combines un-structured vector search (FAISS/mxbai-embed-large) with structured graph queries (Neo4j) via entity extraction to capture both semantic meaning and relational context.
* **Production-Grade Reranking:** Integrates BAAI Cross-Encoder reranking with customizable score normalization (MinMax/Sigmoid/Softmax) to refine hybrid search results.
* **Advanced Query Optimization:** Utilizes HyDE (Hypothetical Document Embeddings) to expand sparse user queries into context-rich semantic search vectors.
* **Multi-Tier Caching Architecture:** Features a FAISS-backed Semantic Cache (SentenceTransformers) for instant  query hits and an SQLite-backed embedding cache to minimize redundant compute.
* **Automated Watcher-Based Ingestion:** Features decoupled background processes for atomic, non-blocking asynchronous ingestion of documents into both vector and graph databases.
* **LLM Tool Calling:** Employs explicit tool binding (e.g., a `numexpr` calculator) to prevent LLM arithmetic hallucinations.

## System Architecture

```text
[ User Input ] 
      │ (via Chainlit UI / REST API)
      ▼
┌────────────────────────────────────────────────────────────┐
│                  FastAPI Backend Server                    │
│                                                            │
│  [ Semantic Cache (FAISS) ] ── (Hit) ──> [ Return Answer ] │
│         │ (Miss)                                           │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ LangGraph Agentic Workflow                           │  │
│  │                                                      │  │
│  │ 1. Router (Vector vs. ChitChat)                      │  │
│  │ 2. Query Expander (HyDE)                             │  │
│  │ 3. Hybrid Retriever (Neo4j + FAISS)                  │  │
│  │ 4. Document Grader (LLM-based relevance check)       │  │
│  │ 5. Query Transformer (Rewrite if docs fail)          │  │
│  │ 6. Tool Executor (Calculator, etc.)                  │  │
│  │ 7. Final Generator (Ollama/phi4-mini)                │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
      │
      ▼
[ Output Response + Citations ]

```

## Tech Stack

| Category | Technology |
| --- | --- |
| **API & Backend** | FastAPI, Uvicorn, Python 3.10+ |
| **AI/Orchestration** | LangChain, LangGraph, LangChain-Experimental |
| **LLMs & Embeddings** | Ollama (`phi4-mini`, `mxbai-embed-large`), SentenceTransformers |
| **Databases** | Neo4j (Graph), FAISS (Vector), SQLite (Memory/Metadata/Cache) |
| **Reranker** | Cross-Encoder (`BAAI/bge-reranker-v2-m3`) |
| **Frontend** | Chainlit |
| **Utilities** | Pydantic, NumExpr (Safe Math), PyPDF2, Tiktoken |

## Project Structure

```text
├── backend/
│   ├── agents/          # LangGraph state machine & LLM agents
│   ├── core/            # System config, logging, and custom exceptions
│   ├── db/              # Local storage for SQLite, FAISS indices, and Neo4j volumes
│   ├── models/          # Pydantic request/response schemas
│   ├── scripts/         # Daemon watchers for atomic file ingestion (Graph & Vector)
│   ├── services/        # Business logic (Retrieval, Memory, Caching, Neo4j connection)
│   ├── tools/           # Bound LLM tools (Calculator, HyDE Expander, Reranker)
│   └── main.py          # FastAPI application entry point
├── frontend/
│   └── chainlit_app.py  # Interactive chat UI
├── docker-compose.yaml  # Containerized Neo4j database setup
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables configuration

```

## Getting Started

### Prerequisites

* Python 3.10+
* Docker & Docker Compose (for Neo4j)
* [Ollama](https://ollama.com/) installed locally with `phi4-mini` and `mxbai-embed-large` models pulled.

### Installation & Setup

1. **Clone the repository and install dependencies:**
```bash
git clone <repo-url>
cd <repo-dir>
pip install -r requirements.txt

```


2. **Start the Neo4j Database:**
```bash
docker-compose up -d

```


3. **Configure the Environment:**
Review and adjust the `.env` file for your local environment (defaults are provided for a local Ollama/Neo4j setup).
4. **Start the Watcher Scripts (Optional but recommended for document ingestion):**
```bash
python backend/scripts/ingest_vector_watch.py --watch knowledge
python backend/scripts/ingest_graph_watch.py --watch knowledge

```


5. **Run the Application:**
* **Backend API:** `uvicorn backend.main:APP --port 8000`
* **Frontend UI:** `chainlit run frontend/chainlit_app.py --port 8001`



## Configuration

Core system behavior is controlled via `.env` and mapped through `backend/core/config.py` using Pydantic Settings. Notable parameters include:

* `MAX_ITERATIONS`: Controls the LangGraph recursion limit for retrieval retries.
* `TOP_K_RETRIEVAL`: Base number of documents to retrieve before reranking.
* `RERANKER_ENABLED` / `USE_HYDE`: Feature flags for advanced retrieval optimizations.
* `SEMANTIC_CACHE_THRESHOLD`: Cosine similarity cutoff (default: `0.80`) for triggering an instant cache hit.

## How It Works — Technical Deep Dive

The retrieval architecture addresses the "lost in the middle" and context-fragmentation problems by utilizing a **Hybrid Strategy**. When a query enters the system, the `RetrieveService` executes two parallel workflows:

1. **Unstructured Semantic Search:** The query is embedded (with an optional HyDE expansion) and run against a FAISS index to find dense semantic matches.
2. **Structured Graph Traversal:** Simultaneously, an LLM extracts named entities (Person, Organization) from the user's query. These entities are passed into a Cypher query against Neo4j to pull multi-hop neighbor relationships (e.g., finding the indirect connection between two isolated facts across different documents).

The merged candidate documents are then passed through a **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`). The reranker computes a precise attention matrix over the `[Query, Document]` pairs, normalizes the logits using MinMax scaling, and truncates the list to the `TOP_K`. Finally, the `GraphRAGAgent` evaluates these top documents. If the grader determines the documents lack the necessary context to answer the user, the query is rewritten by an LLM and the cycle repeats up to `MAX_ITERATIONS`.

## Example Usage

**cURL Request to the REST API:**

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What are the financial implications of the new AI product launch?",
           "top_k": 5,
           "max_tokens": 1024,
           "temperature": 0.1
         }'

```

**Response:**

```json
{
  "query": "What are the financial implications of the new AI product launch?",
  "answer": "Based on the retrieved structured relationships and corporate filings, the new AI product is projected to increase operating margins by 14% while requiring an initial capital expenditure of $4.2M...",
  "sources": [...],
  "num_sources": 5,
  "metadata": {"steps": ["router", "retrieve", "grade_documents", "generate"]}
}

```

## Performance & Design Decisions

* **Thread-Safe Conversational Memory:** The `MemoryService` utilizes `threading.RLock()` to ensure thread safety across concurrent FastAPI requests while asynchronously persisting chat history to an SQLite WAL-mode database.
* **Semantic Caching:** To drastically reduce LLM inference costs and latency, a FAISS-backed semantic cache intercepts queries using an `IndexFlatIP` (Inner Product) similarity search. Highly similar queries ( 0.80 threshold) bypass the agentic workflow entirely, returning cached answers in milliseconds.
* **Atomic Writes & Resilient Ingestion:** The multi-document ingestion scripts (`ingest_multi_docs.py`) write to temporary FAISS indices and JSONL files before performing atomic OS-level file replacements, ensuring the active RAG system never reads corrupted data mid-ingestion.

## Future Improvements

1. **Graph Community Detection:** Implement hierarchical community clustering (similar to Microsoft's GraphRAG approach) to enable global map-reduce summarization capabilities for broad questions like "What is the overall theme of the dataset?".
2. **Async Database Drivers:** Migrate Neo4j and SQLite connections to fully asynchronous drivers (`neo4j.AsyncGraphDatabase` and `aiosqlite`) to maximize FastAPIs event loop efficiency under heavy concurrent load.
3. **Multi-Tenant Architecture:** Expand the SQLite memory and Vector stores to include explicit `user_id` and `tenant_id` partitioning for SaaS deployment scalability.
4. **Dynamic Chunking Strategies:** Transition from static token overlap chunking to semantic chunking (splitting on proposition or embedding shift boundaries) to preserve context integrity during vectorization.

## Author

**Shreyash Gaur** Gen AI & Machine Learning Engineer

[LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/shreyashgaur/) | shreyashgaur221@gmail.com / shreyashgaur01@gmail.com