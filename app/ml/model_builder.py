"""
Model architecture — EfficientNetB0 backbone (ImageNet weights) with a
fresh classification head for the 7 HAM10000 classes, per
00_PROJECT_BLUEPRINT.md training strategy.
"""

from tensorflow import keras
from tensorflow.keras import layers

from app.config import IMG_SIZE, NUM_CLASSES, DEFAULT_LEARNING_RATE


def build_model(learning_rate: float = DEFAULT_LEARNING_RATE, fine_tune_base: bool = False) -> keras.Model:
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    base.trainable = fine_tune_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # EfficientNet expects 0-255 range internally normalized; our images are
    # already scaled to [0,1] in preprocessing.py, so re-scale up before the
    # backbone's own preprocessing layer.
    x = layers.Rescaling(255.0)(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=fine_tune_base)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="dermascan_efficientnet_b0")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
