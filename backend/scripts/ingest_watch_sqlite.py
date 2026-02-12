# backend/scripts/ingest_watch_sqlite.py
import time
import sys
import os
import subprocess
from pathlib import Path
import argparse

# --- CONFIGURATION ---
WATCH_DIR = "knowledge"
FAISS_INDEX_PATH = "backend/db/vector_data/knowledge_faiss.index"
META_JSONL_PATH = "backend/db/vector_data/knowledge_meta.jsonl"
META_DB_PATH = "backend/db/vector_data/metadata_store.db" 
CACHE_DB_PATH = "backend/db/embedding_cache/embed_cache.sqlite"

def ingest_file(file_path: Path):
    print(f"--> Ingesting {file_path.name}...")
    
    # ---------------------------------------------------------
    # STEP 1: Ingest into FAISS + JSONL (The heavy lifting)
    # ---------------------------------------------------------
    ingest_script = Path("backend/scripts/ingest_multi_docs.py")
    
    cmd_list = [
        sys.executable,
        str(ingest_script),
        "--input", str(file_path),
        "--out-index", FAISS_INDEX_PATH,
        "--out-meta", META_JSONL_PATH,
        "--cache-db", CACHE_DB_PATH,
        "--chunk-tokens", "512",
        "--overlap", "128",
        "--batch", "32",
        "--append"
    ]
    
    # We need to make sure PYTHONPATH includes the repo root
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[2]) # ../../
    env["PYTHONPATH"] = repo_root

    result = subprocess.run(cmd_list, env=env)
    
    if result.returncode != 0:
        print(f"!! Error ingesting {file_path.name}")
        return False

    # ---------------------------------------------------------
    # STEP 2: Sync JSONL -> SQLite (The new automatic step)
    # ---------------------------------------------------------
    print(f"--> Syncing metadata to SQLite...")
    sync_script = Path("backend/scripts/convert_meta_to_sqlite.py")

    sync_cmd = [
        sys.executable,
        str(sync_script),
        "--meta", META_JSONL_PATH,
        "--out", META_DB_PATH
    ]

    sync_result = subprocess.run(sync_cmd, env=env)
    
    if sync_result.returncode == 0:
        print(f"✅ Successfully ingested & synced {file_path.name}")
        return True
    else:
        print(f"!! Error syncing SQLite for {file_path.name}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", default=WATCH_DIR, help="Directory to watch")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval (seconds)")
    args = parser.parse_args()

    watch_dir = Path(args.watch)
    watch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Watching {watch_dir} (polling every {args.interval}s)")
    
    # Track known files to avoid re-ingesting on startup
    known_files = set(f.name for f in watch_dir.glob("*") if f.is_file())
    print(f"Current files (ignored): {list(known_files)}")

    while True:
        time.sleep(args.interval)
        current_files = {f.name for f in watch_dir.glob("*") if f.is_file()}
        
        new_files = current_files - known_files
        
        for fname in new_files:
            print(f"New file detected: {fname}")
            full_path = watch_dir / fname
            if ingest_file(full_path):
                known_files.add(fname)
            else:
                # If it failed, don't add to known_files so we retry? 
                # Or add it so we don't loop crash? 
                # Usually safer to add it to avoid infinite loops on bad files.
                known_files.add(fname)

if __name__ == "__main__":
    main()