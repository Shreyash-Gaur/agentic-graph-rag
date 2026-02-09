#!/usr/bin/env python3
"""
Simple folder watcher for auto-ingest.

Polls a directory every N seconds and ingests newly added files by running
the multi-doc ingest script (ingest_multi_docs.py).

Usage:
  python backend/scripts/ingest_watch.py --watch knowledge --interval 10
"""
from __future__ import annotations
import argparse, time, sys, os, shlex, subprocess
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

def run_ingest_file(path: Path):
    """
    Call ingest_multi_docs.py for the given file using subprocess.
    Uses --append to ensure we add to the existing index rather than overwriting it.
    """
    repo_root = Path(__file__).resolve().parents[2]
    
    # We use the same index/meta paths defined in your .env / config
    # (Defaults to backend/db/knowledge_faiss.index)
    cmd_list = [
        sys.executable,
        str(repo_root / "backend" / "scripts" / "ingest_multi_docs.py"),
        "--input", str(path.resolve()),
        "--out-index", "backend/db/knowledge_faiss.index",
        "--out-meta", "backend/db/knowledge_meta.jsonl",
        "--chunk-tokens", "512",
        "--overlap", "128",
        "--batch", "32",
        "--append"  # Critical: Append to existing index
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    
    print(f"--> Ingesting {path.name}...")
    # Using check=False so it doesn't crash the watcher if one file fails
    # We allow stdout/stderr to print to the console so you can see errors
    try:
        subprocess.run(cmd_list, env=env, check=True)
        print(f"--> Done ingesting {path.name}.")
    except subprocess.CalledProcessError as e:
        print(f"!! Error ingesting {path.name}: {e}")

def find_files(dirpath: Path):
    exts = [".pdf", ".txt"]
    out = []
    if not dirpath.exists():
        return []
    for p in dirpath.glob("*"):
        if p.suffix.lower() in exts and p.is_file():
            out.append(p)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", default="knowledge")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    watch_dir = Path(args.watch)
    watch_dir.mkdir(parents=True, exist_ok=True)
    
    # Initial scan: assume existing files are already handled or user will manually trigger
    # (Modify this logic if you want to ingest everything on startup)
    seen = set(p.name for p in find_files(watch_dir))
    print(f"Watching {watch_dir} (polling every {args.interval}s)")
    print(f"Current files (ignored): {sorted(seen)}")
    
    try:
        while True:
            time.sleep(args.interval)
            current_files = find_files(watch_dir)
            current_names = set(p.name for p in current_files)
            
            added = current_names - seen
            
            if added:
                for name in sorted(added):
                    p = watch_dir / name
                    print("New file detected:", p)
                    run_ingest_file(p)
                seen = current_names
    except KeyboardInterrupt:
        print("\nWatcher stopped.")

if __name__ == "__main__":
    main()