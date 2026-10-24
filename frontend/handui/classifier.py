"""Custom gesture samples, training and inference.

Samples are stored as raw landmarks (not features) so the feature function can
change later without re-recording, and so training can augment them.
"""
from __future__ import annotations

import csv
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F
from .config import CLASSIFIER_PATH, SAMPLES_PATH

HEADER = ["label", "handedness"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]
Sample = tuple[str, str, np.ndarray]  # (label, handedness, landmarks)


# ------------------------------------------------------------------- storage

def _write(path: Path, samples: list[Sample], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = mode == "w" or not path.exists()
    with path.open(mode, newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        for label, handedness, lm in samples:
            writer.writerow([label, handedness, *np.round(lm.ravel(), 3)])


def append_samples(label: str, samples: list[tuple[np.ndarray, str]], path: Path = SAMPLES_PATH) -> int:
    _write(path, [(label, handedness, lm) for lm, handedness in samples], "a")
    return len(samples)


def load_samples(path: Path = SAMPLES_PATH) -> list[Sample]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return [(row[0], row[1], np.array(row[2:], dtype=np.float32).reshape(21, 3)) for row in reader]


def label_counts(path: Path = SAMPLES_PATH) -> Counter:
    return Counter(label for label, _, _ in load_samples(path))


def remove_label(label: str, path: Path = SAMPLES_PATH) -> int:
    samples = load_samples(path)
    keep = [s for s in samples if s[0] != label]
    if len(keep) != len(samples):
        _write(path, keep, "w")
    return len(samples) - len(keep)


# ------------------------------------------------------------------ training

def _augment(lm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = F.rotate_landmarks(lm, rng.uniform(-0.2, 0.2))  # about +/-11 degrees
    return out + rng.normal(0.0, 0.012 * F.palm_size(lm), lm.shape).astype(np.float32)


def build_dataset(samples: list[Sample], augment: int = 0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, handedness, lm in samples:
        X.append(F.gesture_features(lm, handedness))
        y.append(label)
        for _ in range(augment):
            X.append(F.gesture_features(_augment(lm, rng), handedness))
            y.append(label)
    return np.array(X), np.array(y)


def _make_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=800, random_state=seed),
    )


def _fit(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    return model


def train(samples_path: Path = SAMPLES_PATH, model_path: Path = CLASSIFIER_PATH,
          augment: int = 3, seed: int = 0) -> dict:
    """Train, report accuracy on held-out data, then retrain on everything and save.

    The last 20% of each label's samples (the end of each recording) are held
    out. Neighbouring frames are near-duplicates, so a random split would
    overstate accuracy.
    """
    samples = load_samples(samples_path)
    counts = Counter(label for label, _, _ in samples)
    if len(counts) < 2:
        raise ValueError(
            "Need samples for at least two labels. Tip: record a 'none' label "
            "with relaxed, random hand poses so normal movement isn't misread."
        )

    by_label: dict[str, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample[0], []).append(sample)
    train_set, test_set = [], []
    for group in by_label.values():
        cut = max(1, int(len(group) * 0.8))
        train_set += group[:cut]
        test_set += group[cut:]

    accuracy = None
    if test_set:
        model = _fit(_make_model(seed), *build_dataset(train_set, augment, seed))
        X_test, y_test = build_dataset(test_set)
        accuracy = float((model.predict(X_test) == y_test).mean())

    model = _fit(_make_model(seed), *build_dataset(samples, augment, seed))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": sorted(counts), "feature_size": F.FEATURE_SIZE}, model_path)
    return {"counts": dict(counts), "holdout_accuracy": accuracy, "model_path": str(model_path)}


# ----------------------------------------------------------------- inference

class GestureClassifier:
    def __init__(self, model, labels: list[str]):
        self.model = model
        self.labels = labels

    @classmethod
    def load(cls, path: Path = CLASSIFIER_PATH) -> "GestureClassifier | None":
        """Load a model this project trained (joblib files are pickles; only load your own)."""
        if not path.exists():
            return None
        bundle = joblib.load(path)
        if bundle.get("feature_size") != F.FEATURE_SIZE:
            print("Gesture classifier was trained with different features; run `python -m handui train` again.")
            return None
        return cls(bundle["model"], bundle["labels"])

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        proba = self.model.predict_proba(features[None, :])[0]
        i = int(np.argmax(proba))
        return str(self.model.classes_[i]), float(proba[i])
