"""
Dataset Manager business logic — image upload/storage, per-class stats,
and the "ready to train?" check used by the AI Studio dashboard.
"""

import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import (
    DATASET_DIR, CLASS_INFO, CLASS_NAMES, ALLOWED_IMAGE_EXTENSIONS,
    MIN_IMAGES_PER_CLASS_TO_TRAIN,
)
from app.models_db import DatasetImage


def save_uploaded_images(db: Session, class_label: str, files: list[UploadFile]):
    if class_label not in CLASS_NAMES:
        raise ValueError(f"Unknown class_label '{class_label}'. Must be one of {CLASS_NAMES}.")

    class_dir = DATASET_DIR / class_label
    class_dir.mkdir(parents=True, exist_ok=True)

    uploaded, skipped, details = 0, 0, []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            skipped += 1
            details.append(f"Skipped '{f.filename}': unsupported extension.")
            continue

        stored_name = f"{uuid.uuid4().hex}{ext}"
        dest_path = class_dir / stored_name

        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        record = DatasetImage(
            filename=f.filename,
            class_label=class_label,
            filepath=str(dest_path),
            source="manual_upload",
        )
        db.add(record)
        uploaded += 1
        details.append(f"Uploaded '{f.filename}' -> {class_label}/{stored_name}")

    db.commit()
    return uploaded, skipped, details


def get_dataset_stats(db: Session):
    counts = dict(
        db.query(DatasetImage.class_label, func.count(DatasetImage.id))
        .group_by(DatasetImage.class_label)
        .all()
    )

    per_class = []
    for cls in CLASS_NAMES:
        info = CLASS_INFO[cls]
        per_class.append({
            "class_label": cls,
            "short_label": info["short_label"],
            "full_label": info["label"],
            "image_count": counts.get(cls, 0),
        })

    total = sum(counts.values())
    ready = all(counts.get(cls, 0) >= MIN_IMAGES_PER_CLASS_TO_TRAIN for cls in CLASS_NAMES)

    return {
        "total_images": total,
        "per_class": per_class,
        "ready_to_train": ready,
        "min_images_per_class_required": MIN_IMAGES_PER_CLASS_TO_TRAIN,
    }


def delete_image(db: Session, image_id: int) -> bool:
    record = db.query(DatasetImage).get(image_id)
    if record is None:
        return False
    path = Path(record.filepath)
    if path.exists():
        path.unlink()
    db.delete(record)
    db.commit()
    return True


def sync_dataset_folder(db: Session):
    """
    Scan data/dataset/<class_label>/* on disk and register any image that
    isn't already tracked in the database yet.

    This is the fast path for large datasets (e.g. the full HAM10000):
    instead of uploading thousands of files one-by-one through the web
    form, just copy/move them straight into data/dataset/<class_label>/
    on disk (e.g. via file explorer, `cp`, or a script), then call this
    (button "Sync Dataset" in the AI Studio, or POST /api/dataset/sync).
    """
    known_paths = {row[0] for row in db.query(DatasetImage.filepath).all()}

    registered, skipped_unknown_class, details = 0, 0, []

    for class_label in CLASS_NAMES:
        class_dir = DATASET_DIR / class_label
        if not class_dir.exists():
            continue
        for f in class_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                continue
            if str(f) in known_paths:
                continue  # already tracked

            record = DatasetImage(
                filename=f.name,
                class_label=class_label,
                filepath=str(f),
                source="folder_sync",
            )
            db.add(record)
            registered += 1
            details.append(f"Registered '{f.name}' -> {class_label}")

    # Also flag any stray class-named folders that don't match a known class,
    # so typos (e.g. "mel " with a trailing space) don't silently vanish.
    for stray in DATASET_DIR.iterdir():
        if stray.is_dir() and stray.name not in CLASS_NAMES:
            skipped_unknown_class += 1
            details.append(f"WARNING: folder '{stray.name}' does not match any known class, ignored.")

    db.commit()
    return registered, skipped_unknown_class, details
