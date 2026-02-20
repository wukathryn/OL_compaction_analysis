# src/init_notebook.py
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))