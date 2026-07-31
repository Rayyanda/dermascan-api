"""
Training endpoints — AI Studio: start a manual training run, poll status,
view history, and inspect a completed run's evaluation results.
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import TrainingStartRequest, TrainingRunOut, TrainingRunDetail
from app.services import dataset_service, training_service
from app.models_db import TrainingRun

router = APIRouter(prefix="/api/training", tags=["Training"])


@router.post("/start", response_model=TrainingRunOut)
def start_training(payload: TrainingStartRequest, db: Session = Depends(get_db)):
    stats = dataset_service.get_dataset_stats(db)
    if not stats["ready_to_train"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset not ready. Every class needs at least "
                f"{stats['min_images_per_class_required']} images before training can start."
            ),
        )

    run = training_service.start_training(
        db, payload.epochs, payload.batch_size, payload.learning_rate
    )
    return run


@router.get("/history", response_model=list[TrainingRunOut])
def training_history(db: Session = Depends(get_db)):
    return db.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()


@router.get("/{run_id}", response_model=TrainingRunDetail)
def training_detail(run_id: int, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found.")

    detail = TrainingRunDetail.model_validate(run)
    if run.classification_report_json:
        detail.classification_report = json.loads(run.classification_report_json)
    if run.confusion_matrix_json:
        detail.confusion_matrix = json.loads(run.confusion_matrix_json)
    if run.history_json:
        detail.history = json.loads(run.history_json)
    return detail
