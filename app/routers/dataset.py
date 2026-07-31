"""
Dataset Manager endpoints — AI Studio: upload images per class, view
per-class statistics, delete a mistakenly-uploaded image.
"""

from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    DatasetStatsResponse, DatasetUploadResponse, DatasetImageOut, DatasetSyncResponse,
)
from app.services import dataset_service
from app.models_db import DatasetImage
from app.config import CLASS_NAMES

router = APIRouter(prefix="/api/dataset", tags=["Dataset Manager"])


@router.get("/classes")
def list_classes():
    """The 7 fixed HAM10000 classes — used to populate upload dropdowns."""
    return {"classes": CLASS_NAMES}


@router.post("/upload", response_model=DatasetUploadResponse)
def upload_images(
    class_label: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    try:
        uploaded, skipped, details = dataset_service.save_uploaded_images(db, class_label, files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return DatasetUploadResponse(uploaded=uploaded, skipped=skipped, details=details)


@router.post("/sync", response_model=DatasetSyncResponse)
def sync_dataset(db: Session = Depends(get_db)):
    """
    For large datasets: place images directly into
    data/dataset/<class_label>/ on disk (file explorer, `cp`, a script,
    etc.) instead of uploading one by one, then call this endpoint to
    register everything new that's sitting in those folders.
    """
    registered, unknown_folders, details = dataset_service.sync_dataset_folder(db)
    return DatasetSyncResponse(
        registered=registered, unknown_class_folders=unknown_folders, details=details
    )


@router.get("/stats", response_model=DatasetStatsResponse)
def dataset_stats(db: Session = Depends(get_db)):
    return dataset_service.get_dataset_stats(db)


@router.get("/images", response_model=List[DatasetImageOut])
def list_images(class_label: str | None = None, db: Session = Depends(get_db)):
    query = db.query(DatasetImage)
    if class_label:
        query = query.filter(DatasetImage.class_label == class_label)
    return query.order_by(DatasetImage.uploaded_at.desc()).limit(500).all()


@router.delete("/images/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    ok = dataset_service.delete_image(db, image_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Image not found.")
    return {"message": "Image deleted."}
