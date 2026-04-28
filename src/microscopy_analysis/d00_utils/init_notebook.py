"""Put the repository's ``src`` directory on ``sys.path`` for notebook use.

Provides a small fallback for notebook environments that have not run
``pip install -e .``. After ``pip install``, the package is importable
without this helper.
"""

import sys
from pathlib import Path


def add_repo_src_to_path() -> Path:
    """Ensure the repository's src directory is importable in notebooks."""
    src_path = Path(__file__).resolve().parents[2]
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src_path


SRC_PATH = add_repo_src_to_path()
