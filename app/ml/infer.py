"""
Inference — used by the Prediction API that the Flutter app calls.

If a model has been trained and deployed via the AI Studio, it is loaded
(and cached in memory) and used for real predictions. If nothing has been
deployed yet, a clearly-flagged dummy prediction is returned instead, so
the Flutter app <-> backend integration can be built and tested end-to-end
before the first model ever finishes training.
"""

import random
import threading

import numpy as np

from app.config import CLASS_NAMES
from app.ml.preprocessing import load_and_preprocess_image, to_batch

_lock = threading.Lock()
_cached_model = None
_cached_model_path = None


def _get_model(model_path: str):
    """Load (and cache) the Keras model for the given path. Thread-safe."""
    global _cached_model, _cached_model_path
    with _lock:
        if _cached_model_path != model_path:
            from tensorflow import keras  # local import: keep TF off the hot path
            _cached_model = keras.models.load_model(model_path)
            _cached_model_path = model_path
        return _cached_model


def clear_model_cache():
    """Call this after a new model is deployed so the old one is dropped."""
    global _cached_model, _cached_model_path
    with _lock:
        _cached_model = None
        _cached_model_path = None


def predict_image(image_path: str, deployed_model_path: str | None):
    """
    Returns (predicted_class: str, probabilities: dict[str, float], is_dummy: bool)
    """
    if deployed_model_path is None:
        return _dummy_predict()

    model = _get_model(deployed_model_path)
    arr = load_and_preprocess_image(image_path)
    batch = to_batch(arr)
    probs = model.predict(batch, verbose=0)[0]

    prob_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    predicted_class = max(prob_dict, key=prob_dict.get)
    return predicted_class, prob_dict, False


def _dummy_predict():
    """
    Deterministic-looking but random dummy output, used only until the
    first model is trained and deployed. Always clearly flagged via
    is_dummy_prediction=True in the API response — never silently mixed
    in with real predictions.
    """
    raw = [random.random() for _ in CLASS_NAMES]
    total = sum(raw)
    probs = [r / total for r in raw]
    # Push one class up so results look plausible/confident rather than flat.
    boosted_idx = random.randrange(len(CLASS_NAMES))
    probs[boosted_idx] += 0.4
    total = sum(probs)
    probs = [p / total for p in probs]

    prob_dict = {cls: p for cls, p in zip(CLASS_NAMES, probs)}
    predicted_class = max(prob_dict, key=prob_dict.get)
    return predicted_class, prob_dict, True
