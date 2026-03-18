"""
Graph Ingestion Watcher
Polls a directory and automatically ingests new files into Neo4j.

Full ingestion pipeline per file:
  A. Load document (PDF or TXT)
  B. Semantic chunking — SemanticChunker splits on meaning boundaries,
     not fixed token counts, so each chunk fed to the LLM is a complete thought
  C. Initialise services (Neo4j, LLM, GraphService)
  D. Coreference resolution (Tier 3) — LLM rewrites each chunk replacing
     pronouns and partial names with full canonical forms before extraction
  E. Entity extraction (Tier 1) — LLMGraphTransformer with strict disambiguation
     prompt extracts entity nodes and typed relationship edges
  F. Write graph structure to Neo4j via add_graph_documents()
  G. Write document embeddings to Neo4j vector store via index_documents_to_neo4j()
     — this is what enables Neo4j hybrid BM25 + dense vector search at query time
  H. Create fulltext index for entity lookup (IF NOT EXISTS — idempotent)
  I. APOC entity merge (Tier 2) — Jaro-Winkler deduplication of formatting variants

Usage:
  python backend/scripts/ingest_graph_watch.py --watch knowledge
  python backend/scripts/ingest_graph_watch.py --watch knowledge --skip-coref
"""
import sys
import time
import argparse
import traceback
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from backend.core.config import settings
from backend.services.graph_service import GraphService

load_dotenv()


def ingest_file(file_path: Path, skip_coref: bool = False):
    print(f"\n--- [Graph] Processing: {file_path.name} ---")

    try:
        # ── A. Load document ─────────────────────────────────────────────────
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path))

        docs = loader.load()

        # ── B. Semantic chunking ─────────────────────────────────────────────
        # SemanticChunker embeds every sentence and splits where cosine
        # similarity between adjacent sentences drops below the percentile
        # threshold — meaning splits happen at actual topic boundaries, not
        # at arbitrary token counts.
        #
        # This directly improves entity extraction quality: LLMGraphTransformer
        # reads one chunk at a time. If a coherent idea about an entity is split
        # across two fixed-size chunks, the LLM sees incomplete context in each
        # and may extract wrong or partial relationships from both.

        print("Chunking document semantically...")
        embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        text_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=settings.SEMANTIC_CHUNK_THRESHOLD_TYPE,
            breakpoint_threshold_amount=settings.SEMANTIC_CHUNK_BREAKPOINT,
        )
        documents = text_splitter.split_documents(docs)
        print(f"Created {len(documents)} semantic chunks.")

        # ── C. Initialise services ───────────────────────────────────────────
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
        )

        llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)

        # GraphService carries all three disambiguation tiers
        graph_service = GraphService()

        # ── D. Coreference resolution (Tier 3) ───────────────────────────────
        # Rewrites each chunk replacing pronouns and partial names with full
        # canonical forms before entity extraction runs.
        # "He said..."  -> "M. Hamel said..."
        # "Claude"      -> "Anthropic Claude" (if established in the same chunk)
        #
        # CLI flag takes precedence over .env — useful for one-off fast runs
        # without permanently changing USE_COREF in .env.
        skip_coref = skip_coref or not settings.USE_COREF

        if skip_coref:
            print("Coreference resolution skipped.")
        else:
            print(f"Running coreference resolution on {len(documents)} chunks...")
            resolved_count = 0
            for doc in documents:
                original = doc.page_content
                doc.page_content = graph_service.resolve_coreferences(doc.page_content)
                if doc.page_content != original:
                    resolved_count += 1
            print(
                f"Coreference resolution complete — "
                f"changed {resolved_count}/{len(documents)} chunks."
            )

        # ── E. Entity extraction (Tier 1) ────────────────────────────────────
        # LLMGraphTransformer with strict disambiguation prompt. Instructs the
        # LLM to always use the most specific name form and to keep distinct
        # entities with shared partial names separate.
        print("Extracting graph nodes & relationships (this takes time)...")

        graph_extraction_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are extracting entities and relationships from text
                to build a knowledge graph. Output valid JSON only.

                STRICT NAMING RULES:
                1. Always use the MOST COMPLETE and SPECIFIC form of every entity name.
                2. NEVER merge distinct entities that share a partial name.
                "M. Hamel" and "C. Hamel" are DIFFERENT people.
                "Claude Shannon" and "Anthropic Claude" are DIFFERENT entities.
                3. Preserve initials exactly — they are the primary disambiguators.
                4. Normalise formatting only: fix spacing and casing, nothing else.
                "M hamel" -> "M. Hamel", "m. hamel" -> "M. Hamel".""",
            ),
            (
                "human",
                (
                    "Extract entities and relationships from this text "
                    "using the provided schema.\n\nText: {input}"
                ),
            ),
        ])

        llm_transformer = LLMGraphTransformer(
            llm=llm,
            prompt=graph_extraction_prompt,
        )
        graph_documents = llm_transformer.convert_to_graph_documents(documents)

        # ── F. Write graph structure to Neo4j ────────────────────────────────
        # Writes entity nodes and typed relationship edges only.
        # Does NOT write document embeddings — that is Step G.
        print(f"Saving {len(graph_documents)} graph documents to Neo4j...")
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True,
        )

        # ── G. Write document embeddings to Neo4j vector store ───────────────
        # add_graph_documents() in Step F only writes entity graph structure.
        # This step embeds each document chunk using mxbai-embed-large and
        # writes the embeddings as Document nodes in Neo4j.
        #
        # Without this step, Neo4jVector.from_existing_graph() in retrieve_service
        # finds no embeddings and Neo4j vector search silently returns empty
        # results — every query falls back to FAISS, defeating the purpose of
        # Neo4j as the primary vector store.
        #
        # Neo4jVector.from_documents() creates the 'document_vector_index'
        # hybrid index (BM25 + dense) automatically on first call.
        print(f"Indexing {len(documents)} chunks into Neo4j vector store...")
        graph_service.index_documents_to_neo4j(documents)
        print("Neo4j vector indexing complete.")

        # ── H. Ensure fulltext index ─────────────────────────────────────────
        # Required for structured_retriever's Cypher query to find entity nodes.
        # IF NOT EXISTS makes this idempotent across multiple ingestion runs.
        try:
            graph.query(
                "CREATE FULLTEXT INDEX fulltext_entity_id IF NOT EXISTS "
                "FOR (n:__Entity__) ON EACH [n.id]"
            )
        except Exception as e:
            print(f"  (Fulltext index note: {e})")

        # ── I. APOC entity merge (Tier 2) ────────────────────────────────────
        # Finds and merges entity nodes that are formatting variants of the same
        # real-world entity using Jaro-Winkler similarity at threshold 0.92.
        # "M Hamel" + "M. Hamel"  -> merged into "M. Hamel" (longer = canonical)
        # "M. Hamel" + "C. Hamel" -> kept separate (score ~0.81, below threshold)
        # Requires APOC — configured in docker-compose.yaml.
        print("Running entity deduplication (APOC Jaro-Winkler merge)...")
        merged = graph_service.merge_duplicate_entities()
        if merged > 0:
            print(f"  Merged {merged} duplicate entity pair(s).")
        else:
            print("  No duplicates found — graph is clean.")

        print(f"--- [Graph] Success: {file_path.name} ---")

    except Exception as e:
        print(f"!!! [Graph] Failed to ingest {file_path.name}: {e}")
        traceback.print_exc()


def find_files(dirpath: Path):
    """Return supported files in the watch directory."""
    exts = [".pdf", ".txt", ".md"]
    if not dirpath.exists():
        return []
    return [
        p for p in dirpath.glob("*")
        if p.suffix.lower() in exts and p.is_file()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Watch folder and ingest into Neo4j with entity disambiguation"
    )
    parser.add_argument(
        "--watch", default=settings.WATCH_DIR, help="Folder to watch"
    )
    parser.add_argument(
        "--interval", type=int, default=10, help="Polling interval in seconds"
    )
    parser.add_argument(
        "--skip-coref", action="store_true",
        help=(
            "Skip coreference resolution for this run. Overrides USE_COREF in .env. "
            "Faster ingestion but less accurate entity extraction."
        ),
    )
    args = parser.parse_args()

    watch_dir = Path(args.watch)
    watch_dir.mkdir(parents=True, exist_ok=True)

    # Resolve effective coref setting: CLI flag overrides .env
    effective_coref = not args.skip_coref and settings.USE_COREF

    print("--- Graph Watcher Started ---")
    print(f"Watching:           {watch_dir.resolve()}")
    print(f"Target DB:          {settings.NEO4J_URI}")
    print(f"Chunking:           semantic (type={settings.SEMANTIC_CHUNK_THRESHOLD_TYPE}, breakpoint={settings.SEMANTIC_CHUNK_BREAKPOINT})")
    print(f"Coreference pass:   {'enabled' if effective_coref else 'disabled'}")
    print(f"Entity merge:       APOC Jaro-Winkler (threshold=0.92)")

    current_files = find_files(watch_dir)
    seen = set(p.name for p in current_files)
    print(f"Skipping {len(seen)} existing file(s). Waiting for new files...")

    try:
        while True:
            time.sleep(args.interval)

            current_files = find_files(watch_dir)
            current_names = set(p.name for p in current_files)
            new_file_names = current_names - seen

            if new_file_names:
                for name in sorted(new_file_names):
                    ingest_file(watch_dir / name, skip_coref=args.skip_coref)
                seen = current_names

    except KeyboardInterrupt:
        print("\nWatcher stopped.")


if __name__ == "__main__":
    main()