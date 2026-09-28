"""
TRAFFIQ Cloud Root Entrypoint
Allows deploying directly from repository root on Render, Railway, Fly.io, etc.
"""

import os
import sys
from pathlib import Path

# Add backend directory and root directory to sys.path
_ROOT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _ROOT_DIR / "backend"
for _p in [str(_BACKEND_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from backend.main import app
except ImportError:
    from main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port)
