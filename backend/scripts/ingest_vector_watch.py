# backend/scripts/ingest_vector_watch.py
"""
Simple folder watcher for auto-ingest.

Polls a directory every N seconds and ingests newly added files by running
the multi-doc ingest script (ingest_multi_docs.py) and converts the metadata to SQLite by running the convert_meta_to_sqlite.py script.

Usage:
  python backend/scripts/ingest_vector_watch.py --watch knowledge --interval 10
"""





import time
import sys
import os
import subprocess
from pathlib import Path
import argparse

# --- 1. SETUP & CONFIG IMPORT ---
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.config import settings

# --- CONFIGURATION (Pulled from config.py) ---
FAISS_INDEX_PATH = settings.FAISS_INDEX_PATH
META_JSONL_PATH = settings.FAISS_META_PATH
CACHE_DB_PATH = settings.EMBEDDING_CACHE_DB
META_DB_PATH = settings.META_DB_PATH

# --- 2. FILE FINDER HELPER ---
def find_files(dirpath: Path):
    """
    Returns a list of valid files based on allowed extensions.
    Add '.md', '.csv', etc. to 'exts' here to support more formats.
    """
    exts = [".pdf", ".txt", ".md"]
    out = []
    if not dirpath.exists():
        return []
    
    for p in dirpath.glob("*"):
        # Check extension and ensure it's a file (not a folder)
        if p.suffix.lower() in exts and p.is_file():
            out.append(p)
    return out

# --- 3. INGESTION LOGIC ---
def ingest_file(file_path: Path):
    print(f"--> Ingesting {file_path.name}...")
    
    # A. Ingest into FAISS + JSONL
    ingest_script = Path("backend/scripts/ingest_multi_docs.py")
    
    cmd_list = [
        sys.executable,
        str(ingest_script),
        "--input", str(file_path),
        "--out-index", FAISS_INDEX_PATH,
        "--out-meta", META_JSONL_PATH,
        "--cache-db", CACHE_DB_PATH,
        "--chunk-tokens", str(settings.CHUNK_TOKENS), 
        "--overlap", str(settings.CHUNK_OVERLAP),     
        "--batch", str(settings.EMBEDDING_BATCH_SIZE),
        "--append"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(cmd_list, env=env)
    
    if result.returncode != 0:
        print(f"!! Error ingesting {file_path.name}")
        return False

    # B. Sync JSONL -> SQLite
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

# --- 4. MAIN WATCHER LOOP ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", default=settings.WATCH_DIR, help="Directory to watch")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval (seconds)")
    args = parser.parse_args()

    watch_dir = Path(args.watch)
    watch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Watching {watch_dir} (polling every {args.interval}s)")
    
    # Initial Scan: Mark existing valid files as 'seen'
    # We use find_files() here so we don't accidentally track ignored file types
    known_files = set(f.name for f in find_files(watch_dir))
    print(f"Current valid files (ignored): {list(known_files)}")

    try:
        while True:
            time.sleep(args.interval)
            
            # Get currently valid files
            current_file_objs = find_files(watch_dir)
            current_names = {f.name for f in current_file_objs}
            
            # Identify new files
            new_files = current_names - known_files
            
            for fname in sorted(new_files):
                print(f"New file detected: {fname}")
                full_path = watch_dir / fname
                
                # Run ingestion
                if ingest_file(full_path):
                    known_files.add(fname)
                else:
                    # If it fails, we still add it to 'known_files' 
                    # to prevent an infinite loop of retries every 10s.
                    print(f"Skipping {fname} due to errors.")
                    known_files.add(fname)
                    
            # Handle deletions (optional cleanup of local set)
            # If a file is deleted, we remove it from known_files so 
            # if it's added back later, we re-ingest it.
            deleted_files = known_files - current_names
            if deleted_files:
                for f in deleted_files:
                    known_files.remove(f)
                
    except KeyboardInterrupt:
        print("\nWatcher stopped by user.")

if __name__ == "__main__":
    main()