"""
Loads and prepares the labelled dataset used to train the document
domain classifier.
"""
import json
import os
from typing import List, Tuple

import pandas as pd

from config.settings import settings


def load_dataset(csv_path: str = None) -> pd.DataFrame:
    csv_path = csv_path or os.path.join(settings.DATASET_DIR, "sample_dataset.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    return df


def build_label_maps(df: pd.DataFrame) -> Tuple[dict, dict]:
    labels = sorted(df["label"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def prepare_training_data(csv_path: str = None):
    df = load_dataset(csv_path)
    label2id, id2label = build_label_maps(df)

    texts: List[str] = df["text"].tolist()
    numeric_labels: List[int] = [label2id[l] for l in df["label"].tolist()]

    # Persist label map so predictor.py can decode predictions later.
    os.makedirs(os.path.dirname(settings.LABELS_PATH), exist_ok=True)
    with open(settings.LABELS_PATH, "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    return texts, numeric_labels, label2id, id2label
