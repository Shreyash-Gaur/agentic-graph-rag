# backend/services/debug_memory_run.py
"""
Tiny helper to manually exercise MemoryService from the command-line.

Run:
    python backend/services/debug_memory_run.py
"""

import sys
from pathlib import Path
# Ensure project root on sys.path so `backend` is importable
REPO_ROOT = Path(__file__).resolve().parents[2]  # debug_memory_run.py is backend/services -> parents[2] -> repo root
sys.path.insert(0, str(REPO_ROOT))

import pprint
from backend.services.memory_service import MemoryService

def main():
    m = MemoryService(max_history=5, use_sqlite=True, db_path="backend/db/memory/memory_store.sqlite")
    print("Adding two turns into conversation 'conv1'...")
    m.add_turn("conv1", "user", "Hello")
    m.add_turn("conv1", "assistant", "Hi there")

    print("\nIn-memory history for conv1:")
    pprint.pprint(m.get_history("conv1"))

    print("\nExported history (JSON):")
    print(m.export_history("conv1"))

    print("\nClearing history for conv1...")
    m.clear_history("conv1")
    print("After clear, get_history:", m.get_history("conv1"))

    m.close()
    print("\nDone. DB persisted at backend/db/memory/memory_store.sqlite (if use_sqlite=True)")

if __name__ == "__main__":
    main()
