"""
Graph Ingestion Watcher
Polls a directory and automatically ingests new files into Neo4j.
Usage: python backend/scripts/ingest_graph_watch.py --watch knowledge
"""
import sys
import time
import argparse
import traceback
from pathlib import Path
from dotenv import load_dotenv

# --- 1. Setup Environment & Imports ---
# Ensure repo root is on sys.path so we can import backend modules
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from backend.core.config import settings

load_dotenv()

# --- 2. The Ingestion Logic (From ingest_graph.py) ---
def ingest_file(file_path: Path):
    print(f"\n--- [Graph] Processing: {file_path.name} ---")
    
    try:
        # A. Load & Split
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path))
            
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
        documents = text_splitter.split_documents(documents=docs)
        print(f"Created {len(documents)} chunks.")

        # B. Initialize Neo4j & LLM
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
        
        llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)
        llm_transformer = LLMGraphTransformer(llm=llm)

        # C. Convert to Graph Documents
        print("Extracting graph nodes & relationships (this takes time)...")
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        
        # D. Add to Neo4j
        print(f"Saving {len(graph_documents)} graph documents to Neo4j...")
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        
        # E. Ensure Index
        try:
            graph.query(
                "CREATE FULLTEXT INDEX fulltext_entity_id IF NOT EXISTS FOR (n:__Entity__) ON EACH [n.id]"
            )
        except Exception as e:
            # Ignore if index already exists or other minor issue
            pass

        print(f"--- [Graph] Success: {file_path.name} ---")

    except Exception as e:
        print(f"!!! [Graph] Failed to ingest {file_path.name}: {e}")
        traceback.print_exc()

# --- 3. The Watcher Logic (From ingest_watch.py) ---
def find_files(dirpath: Path):
    """Return a list of supported files in the directory."""
    exts = [".pdf", ".txt", ".md"]
    out = []
    if not dirpath.exists():
        return []
    for p in dirpath.glob("*"):
        if p.suffix.lower() in exts and p.is_file():
            out.append(p)
    return out

def main():
    parser = argparse.ArgumentParser(description="Watch folder and ingest into Neo4j")
    parser.add_argument("--watch", default="knowledge", help="Folder to watch")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds")
    args = parser.parse_args()

    watch_dir = Path(args.watch)
    watch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Graph Watcher Started ---")
    print(f"Watching: {watch_dir.resolve()}")
    print(f"Target DB: {settings.NEO4J_URI}")
    
    # 1. Initial Scan (Mark existing files as 'seen' so we don't re-ingest them all on startup)
    # If you WANT to re-ingest everything on startup, change this to `seen = set()`
    current_files = find_files(watch_dir)
    seen = set(p.name for p in current_files)
    
    print(f"Skipping {len(seen)} existing files (already in folder). Waiting for NEW files...")

    # 2. Polling Loop
    try:
        while True:
            time.sleep(args.interval)
            
            current_files = find_files(watch_dir)
            current_names = set(p.name for p in current_files)
            
            # Calculate what is new
            new_file_names = current_names - seen
            
            if new_file_names:
                for name in sorted(new_file_names):
                    file_path = watch_dir / name
                    # Run the ingestion function directly
                    ingest_file(file_path)
                
                # Update seen set
                seen = current_names
                
    except KeyboardInterrupt:
        print("\nWatcher stopped by user.")

if __name__ == "__main__":
    main()