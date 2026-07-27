"""
Client-side wrapper used by the main FastAPI application: runs document
classification in an isolated subprocess (see _predict_worker.py) instead of
importing TensorFlow directly into the server process. See _predict_worker.py
for the rationale (avoids a native gRPC/protobuf init-order conflict with
ChromaDB observed on this platform).
"""
import json
import subprocess
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def predict_category_isolated(text: str, timeout: int = 60):
    """Returns (category: str, confidence: float). Falls back to
    ('Uncategorized', 0.0) if the worker fails or times out."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.ml._predict_worker", text[:4000]],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        # TensorFlow prints assorted log lines to stdout/stderr; the worker's
        # JSON is always the LAST non-empty stdout line.
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return "Uncategorized", 0.0
        payload = json.loads(lines[-1])
        return payload.get("category", "Uncategorized"), float(payload.get("confidence", 0.0))
    except Exception:
        return "Uncategorized", 0.0
