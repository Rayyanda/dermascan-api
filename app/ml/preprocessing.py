"""
Image preprocessing shared by both the training pipeline and the
single-image inference path, so a prediction always sees images
processed exactly the way the model was trained on.
"""

import numpy as np
from PIL import Image

from app.config import IMG_SIZE


def load_and_preprocess_image(path_or_file) -> np.ndarray:
    """
    Load an image from a path or file-like object, resize to IMG_SIZE,
    convert to RGB, and scale pixels to [0, 1].

    Returns an array of shape (H, W, 3), dtype float32.
    """
    img = Image.open(path_or_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def to_batch(arr: np.ndarray) -> np.ndarray:
    """Add the batch dimension expected by model.predict(): (1, H, W, 3)."""
    return np.expand_dims(arr, axis=0)
