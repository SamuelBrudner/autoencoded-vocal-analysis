from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Ensure repo root is importable for tests that reference src/.
    sys.path.insert(0, str(ROOT))
