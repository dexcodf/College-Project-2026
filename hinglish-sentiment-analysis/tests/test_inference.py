"""
Smoke tests for the inference contract.

These are skipped automatically if no trained model is present, so the suite
stays green on a fresh checkout (run `python -m src.train` to enable them).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, LABELS  # noqa: E402

_MODEL_READY = (CONFIG.best_model_path / "model_meta.json").exists()
pytestmark = pytest.mark.skipif(not _MODEL_READY, reason="No trained model available.")


@pytest.fixture(scope="module")
def predictor():
    from src.inference import get_predictor
    return get_predictor()


def test_predict_returns_valid_schema(predictor):
    r = predictor.predict("Bhai app bahut acha hai")
    assert r["sentiment"] in LABELS
    assert 0.0 <= r["confidence"] <= 1.0
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-3


def test_predict_batch_length(predictor):
    out = predictor.predict_batch(["acha hai", "bakwaas hai"])
    assert len(out) == 2
