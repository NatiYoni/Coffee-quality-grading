"""Shared fixtures: the API is exercised with lightweight fakes standing in for
the trained models, so the suite runs in seconds and needs neither TensorFlow
nor the Git-LFS model artifacts."""

import base64
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402

FEATURE_COLS = [
    "Altitude", "Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
    "Uniformity", "Clean Cup", "Sweetness", "Moisture Percentage",
    "Category One Defects", "Category Two Defects",
]

SENSORY = {"Aroma": 8.2, "Flavor": 8.1, "Acidity": 8.0, "Body": 7.9, "Balance": 8.0}


class FakeRegressor:
    def __init__(self, score):
        self.score = score
        self.seen = None

    def predict(self, X):
        self.seen = np.asarray(X)
        return np.array([self.score])


class FakeExplainer:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def shap_values(self, X):
        return np.tile(self.values, (len(X), 1))


class FakeKMeans:
    def __init__(self, cluster=2):
        self.cluster = cluster
        self.seen = None

    def predict(self, X):
        self.seen = np.asarray(X)
        return np.array([self.cluster])


class IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class FakePreds:
    def __init__(self, probs):
        self._probs = np.asarray([probs], dtype=np.float32)

    def numpy(self):
        return self._probs


class FakeImageModel:
    """Mimics the Keras call signature: model(batch, training=False).numpy()."""

    def __init__(self, probs, size=(224, 224)):
        self.probs = probs
        self.input_shape = (None, size[0], size[1], 3)
        self.seen = None

    def __call__(self, arr, training=False):
        self.seen = arr
        return FakePreds(self.probs)


@pytest.fixture
def client():
    # No context manager: startup (real model loading) must not run.
    return TestClient(main.app)


@pytest.fixture
def unloaded(monkeypatch):
    for name in [
        "xgb_model", "shap_explainer", "kmeans_model", "scaler", "cluster_labels",
        "feature_cols", "roast_model", "roast_classes", "defect_model", "defect_classes",
    ]:
        monkeypatch.setattr(main, name, None)


@pytest.fixture
def loaded(monkeypatch):
    fakes = {
        "xgb_model": FakeRegressor(84.2),
        # Flavor and Balance push up, Altitude pushes down; the rest are zero.
        "shap_explainer": FakeExplainer([-0.3, 0.1, 0.9, 0.0, 0.05, 0.0, 0.6, 0, 0, 0, 0, 0, 0]),
        "kmeans_model": FakeKMeans(cluster=2),
        "scaler": IdentityScaler(),
        "cluster_labels": {
            "0": {"name": "Bright & Floral", "description": "High acidity and aroma"},
            "2": {"name": "Balanced & Sweet", "description": "High balance and sweetness"},
        },
        "feature_cols": list(FEATURE_COLS),
        "roast_model": FakeImageModel([0.05, 0.02, 0.9, 0.03]),
        "roast_classes": ["Dark", "Green", "Light", "Medium"],
        "defect_model": FakeImageModel([0.1, 0.7, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]),
        "defect_classes": {"black": 0, "broken": 1, "foreign": 2, "fraghusk": 3, "green": 4,
                           "husk": 5, "immature": 6, "infested": 7, "sour": 8},
        "regions_data": [{"region": "Guji", "avg_score": 86.0}],
    }
    for name, value in fakes.items():
        monkeypatch.setattr(main, name, value)
    return fakes


def png_base64(size=(32, 20), mode="RGB", color=(120, 80, 40)):
    buf = BytesIO()
    Image.new(mode, size, color if mode == "RGB" else 128).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
