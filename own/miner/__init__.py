"""Own Pearl miner, host side. The reference library ref/ (pearl_ref, blake3_np) sits next to this package."""
import os
import sys
from pathlib import Path

VERSION = "0.2.0"

_REF = Path(os.environ.get("PEARL_REF", Path(__file__).resolve().parent.parent / "ref"))
if not (_REF / "pearl_ref.py").exists():
    raise ImportError(f"pearl_ref.py not found in {_REF} (set PEARL_REF)")
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))
