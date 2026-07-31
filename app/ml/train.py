"""
Training pipeline — runs the full AI workflow from the blueprint:
Dataset Validation -> Preprocessing -> Augmentation -> Split -> Training
-> Evaluation -> Export.

Designed to run inside a background thread (see services/training_service.py),
each run gets its own SQLAlchemy session so it never fights the request
thread's session.
"""

import datetime as dt
import json
import traceback

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from app.config import (
    IMG_SIZE, NUM_CLASSES, CLASS_NAMES, MODELS_DIR, VALIDATION_SPLIT,
)
from app.database import SessionLocal
from app.models_db import DatasetImage, TrainingRun, ModelVersion, TrainingStatus
from app.ml.model_builder import build_model
from app.ml.preprocessing import load_and_preprocess_image


AUGMENTATION = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.08),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
], name="augmentation")


def _build_dataset(paths, labels, batch_size, training: bool) -> tf.data.Dataset:
    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    idx_labels = [label_to_idx[l] for l in labels]
    one_hot = keras.utils.to_categorical(idx_labels, num_classes=NUM_CLASSES)

    def _load(path, label):
        def _py_load(p):
            arr = load_and_preprocess_image(p.numpy().decode("utf-8"))
            return arr.astype(np.float32)
        img = tf.py_function(_py_load, [path], tf.float32)
        img.set_shape((*IMG_SIZE, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, one_hot))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=min(1000, len(paths)))
        ds = ds.map(lambda x, y: (AUGMENTATION(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def run_training(training_run_id: int) -> None:
    """
    Entry point called from a background thread. Loads the dataset from
    disk, trains, evaluates, exports the model, and writes every result
    back onto the TrainingRun / ModelVersion rows.
    """
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).get(training_run_id)
        if run is None:
            return

        run.status = TrainingStatus.RUNNING
        run.started_at = dt.datetime.utcnow()
        db.commit()

        images = db.query(DatasetImage).all()
        paths = [img.filepath for img in images]
        labels = [img.class_label for img in images]

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=VALIDATION_SPLIT, stratify=labels, random_state=42,
        )

        run.num_train_images = len(train_paths)
        run.num_val_images = len(val_paths)
        db.commit()

        train_ds = _build_dataset(train_paths, train_labels, run.batch_size, training=True)
        val_ds = _build_dataset(val_paths, val_labels, run.batch_size, training=False)

        model = build_model(learning_rate=run.learning_rate)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=run.epochs,
            shuffle=False,  # dataset is already shuffled via .shuffle() above
        )

        # --- Evaluation ---------------------------------------------------
        val_probs = model.predict(val_ds)
        val_pred_idx = np.argmax(val_probs, axis=1)
        label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        val_true_idx = [label_to_idx[l] for l in val_labels]

        report = classification_report(
            val_true_idx, val_pred_idx, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(val_true_idx, val_pred_idx).tolist()

        # --- Export ---------------------------------------------------------
        model_filename = f"model_run{training_run_id}_{dt.datetime.utcnow():%Y%m%d%H%M%S}.keras"
        model_path = MODELS_DIR / model_filename
        model.save(model_path)

        # --- Persist results -------------------------------------------------
        run.train_accuracy = float(history.history["accuracy"][-1])
        run.val_accuracy = float(history.history["val_accuracy"][-1])
        run.train_loss = float(history.history["loss"][-1])
        run.val_loss = float(history.history["val_loss"][-1])
        run.classification_report_json = json.dumps(report)
        run.confusion_matrix_json = json.dumps(cm)
        run.history_json = json.dumps({
            k: [float(v) for v in vals] for k, vals in history.history.items()
        })
        run.status = TrainingStatus.COMPLETED
        run.finished_at = dt.datetime.utcnow()
        db.commit()

        model_version = ModelVersion(
            training_run_id=training_run_id,
            filename=model_filename,
            filepath=str(model_path),
            val_accuracy=run.val_accuracy,
            is_deployed=False,
        )
        db.add(model_version)
        db.commit()

    except Exception as exc:  # noqa: BLE001 — we want to capture and store any failure
        db.rollback()
        run = db.query(TrainingRun).get(training_run_id)
        if run is not None:
            run.status = TrainingStatus.FAILED
            run.error_message = f"{exc}\n{traceback.format_exc()}"
            run.finished_at = dt.datetime.utcnow()
            db.commit()
    finally:
        db.close()
