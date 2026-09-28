"""
TRAFFIQ Backend Package
"""
import sys
from pathlib import Path

# Add backend directory and root directory to sys.path
_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent
for _p in [str(_BACKEND_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
