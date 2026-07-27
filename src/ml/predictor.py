"""
Loads the trained classifier (TensorFlow .h5 model, or the scikit-learn
fallback) and exposes a single `predict_category()` function used by the
document ingestion pipeline to auto-tag uploaded PDFs.
"""
import json
import os
import pickle
from functools import lru_cache

from config.settings import settings

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:  # pragma: no cover
    TF_AVAILABLE = False


class DocumentClassifier:
    def __init__(self):
        self.engine = None
        self.model = None
        self.vectorizer = None          # sklearn TF-IDF vectorizer (fallback engine)
        self.tf_vectorize_layer = None  # rebuilt TextVectorization layer (tensorflow engine)
        self.id2label = {}
        self._load()

    def _load(self):
        labels_path = settings.LABELS_PATH
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                label_map = json.load(f)
                self.id2label = {int(k): v for k, v in label_map.get("id2label", {}).items()}

        if TF_AVAILABLE and os.path.exists(settings.MODEL_PATH) and os.path.exists(settings.TOKENIZER_PATH):
            from tensorflow.keras import layers

            self.engine = "tensorflow"
            self.model = tf.keras.models.load_model(settings.MODEL_PATH)

            with open(settings.TOKENIZER_PATH, "rb") as f:
                tok = pickle.load(f)

            self.tf_vectorize_layer = layers.TextVectorization(
                max_tokens=tok["vocab_size"],
                output_mode="int",
                output_sequence_length=tok["max_len"],
                vocabulary=tok["vocabulary"][2:],  # skip reserved "" and "[UNK]" entries
            )
            return

        sklearn_path = settings.MODEL_PATH.replace(".h5", "_sklearn.pkl")
        if os.path.exists(sklearn_path):
            self.engine = "sklearn"
            with open(sklearn_path, "rb") as f:
                artifact = pickle.load(f)
                self.model = artifact["classifier"]
                self.vectorizer = artifact["vectorizer"]
            return

        self.engine = None  # No trained model yet.

    def predict(self, text: str):
        """Returns (category: str, confidence: float 0-1). Falls back to
        'Uncategorized' / 0.0 if no model has been trained yet."""
        if self.engine is None or not text.strip():
            return "Uncategorized", 0.0

        snippet = text[:2000]  # classification only needs a representative slice

        if self.engine == "tensorflow":
            import tensorflow as tf
            sequence = self.tf_vectorize_layer(tf.constant([snippet]))
            probs = self.model.predict(sequence, verbose=0)[0]
            idx = int(probs.argmax())
            return self.id2label.get(idx, "Uncategorized"), float(probs[idx])

        if self.engine == "sklearn":
            X = self.vectorizer.transform([snippet])
            probs = self.model.predict_proba(X)[0]
            idx = int(probs.argmax())
            return self.id2label.get(idx, "Uncategorized"), float(probs[idx])

        return "Uncategorized", 0.0


@lru_cache(maxsize=1)
def get_classifier() -> DocumentClassifier:
    """Cached singleton so the model is loaded from disk only once."""
    return DocumentClassifier()
