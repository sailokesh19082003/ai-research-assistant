"""
Trains a deep-learning text classifier (TensorFlow/Keras) that predicts the
technical domain/category of an uploaded document ("Artificial Intelligence",
"Cyber Security", "Cloud Computing", "Robotics", "Blockchain", "Networking").

Architecture (per spec):
  TextVectorization -> Embedding -> GlobalAveragePooling1D
  -> Dense(128, relu) -> Dropout(0.3) -> Dense(num_classes, softmax)

If TensorFlow is unavailable in the runtime environment, the module
transparently falls back to a scikit-learn TF-IDF + Logistic Regression
classifier so the rest of the application (ingestion, RAG, analytics) keeps
working end-to-end. `predictor.py` knows how to load whichever artifact was
produced.
"""
import json
import pickle
import numpy as np

from config.settings import settings
from src.ml.dataset_prep import prepare_training_data

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TF_AVAILABLE = False


def _train_with_tensorflow(texts, labels, num_classes, vocab_size=10000, max_len=200):
    """
    NOTE on architecture: the TextVectorization layer is kept OUTSIDE the
    saved Keras model and its vocabulary is persisted separately to
    tokenizer.pickle. Embedding stateful preprocessing layers directly
    inside a model saved to legacy HDF5 (.h5) format is a known failure
    mode (the string-lookup table doesn't get restored on load), so the
    model itself only ever sees integer token sequences -- this is both
    more robust and matches the spec's `tokenizer.pickle` artifact.
    """
    labels = np.array(labels)

    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
    )
    vectorize_layer.adapt(texts)

    # Persist the fitted vocabulary so predictor.py can rebuild an identical
    # vectorizer at inference time without needing to re-adapt.
    vocabulary = vectorize_layer.get_vocabulary()
    with open(settings.TOKENIZER_PATH, "wb") as f:
        pickle.dump({"vocabulary": vocabulary, "max_len": max_len, "vocab_size": vocab_size}, f)

    # Pre-vectorize all texts to integer sequences up front.
    sequences = vectorize_layer(tf.constant(texts)).numpy()

    model = models.Sequential([
        layers.Input(shape=(max_len,), dtype="int64"),
        layers.Embedding(vocab_size, 64, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    n_val = max(1, int(len(sequences) * 0.2))
    train_x, val_x = sequences[:-n_val], sequences[-n_val:]
    train_y, val_y = labels[:-n_val], labels[-n_val:]

    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=15,
        batch_size=4,
        verbose=2,
    )

    model.save(settings.MODEL_PATH)
    return model, history.history


def _train_with_sklearn(texts, labels, num_classes):
    """Fallback classifier: TF-IDF vectorizer + Logistic Regression.

    Kept deliberately simple/interpretable; swapped in automatically only
    when TensorFlow cannot be imported in the current environment.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, clf.predict(X_val))

    fallback_model_path = settings.MODEL_PATH.replace(".h5", "_sklearn.pkl")
    with open(fallback_model_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "classifier": clf}, f)

    return {"model_path": fallback_model_path, "val_accuracy": val_acc}


def build_and_train_classifier(csv_path: str = None):
    texts, numeric_labels, label2id, id2label = prepare_training_data(csv_path)
    num_classes = len(label2id)

    if TF_AVAILABLE:
        print("[train_classifier] TensorFlow detected -> training Keras model.")
        model, history = _train_with_tensorflow(texts, numeric_labels, num_classes)
        result = {
            "engine": "tensorflow",
            "model_path": settings.MODEL_PATH,
            "final_train_accuracy": history["accuracy"][-1],
            "final_val_accuracy": history["val_accuracy"][-1],
        }
    else:
        print("[train_classifier] TensorFlow unavailable -> using scikit-learn fallback.")
        sk_result = _train_with_sklearn(texts, numeric_labels, num_classes)
        result = {
            "engine": "sklearn",
            "model_path": sk_result["model_path"],
            "final_val_accuracy": sk_result["val_accuracy"],
        }

    with open(settings.LABELS_PATH.replace(".json", "_meta.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("[train_classifier] Training complete:", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    build_and_train_classifier()
