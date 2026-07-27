"""
Standalone worker invoked as a subprocess to run TensorFlow classifier
inference in complete isolation from the main FastAPI process.

Why: loading TensorFlow and ChromaDB's native gRPC/protobuf dependencies in
the *same* process has been observed to segfault depending on import/load
order and platform. Running inference in a short-lived child process
sidesteps the issue entirely at a small (~1-2s) latency cost, which is
irrelevant since classification runs inside a background ingestion task.

Usage:
    python -m src.ml._predict_worker "some text to classify"

Prints a single line of JSON to stdout: {"category": ..., "confidence": ...}
"""
import sys
import json


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else ""

    try:
        from src.ml.predictor import DocumentClassifier
        clf = DocumentClassifier()
        category, confidence = clf.predict(text)
        print(json.dumps({"category": category, "confidence": confidence}))
    except Exception as e:
        print(json.dumps({"category": "Uncategorized", "confidence": 0.0, "error": str(e)}))


if __name__ == "__main__":
    main()
