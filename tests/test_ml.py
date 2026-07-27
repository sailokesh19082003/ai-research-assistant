import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.dataset_prep import load_dataset, build_label_maps


def test_dataset_loads():
    df = load_dataset()
    assert len(df) > 0
    assert "text" in df.columns
    assert "label" in df.columns


def test_label_maps_are_consistent():
    df = load_dataset()
    label2id, id2label = build_label_maps(df)
    assert len(label2id) == len(id2label)
    for label, idx in label2id.items():
        assert id2label[idx] == label


def test_classifier_predicts_uncategorized_without_trained_model(tmp_path, monkeypatch):
    from config.settings import settings
    from src.ml.predictor import DocumentClassifier

    monkeypatch.setattr(settings, "MODEL_PATH", str(tmp_path / "nope.h5"))
    monkeypatch.setattr(settings, "LABELS_PATH", str(tmp_path / "nope_labels.json"))

    clf = DocumentClassifier()
    category, confidence = clf.predict("Some arbitrary text.")
    assert category == "Uncategorized"
    assert confidence == 0.0
