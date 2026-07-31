"""
Dashboard endpoint — powers the AI Studio home screen: quick counts and
the current state of the pipeline (dataset size, last training run,
which model is live).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import DashboardSummary
from app.models_db import DatasetImage, TrainingRun, PredictionLog
from app.services import model_service

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def summary(db: Session = Depends(get_db)):
    total_images = db.query(DatasetImage).count()
    total_runs = db.query(TrainingRun).count()
    last_run = db.query(TrainingRun).order_by(TrainingRun.created_at.desc()).first()
    active_model = model_service.get_deployed_model(db)
    total_predictions = db.query(PredictionLog).count()

    return DashboardSummary(
        total_dataset_images=total_images,
        total_training_runs=total_runs,
        active_model=active_model,
        last_training_run=last_run,
        total_predictions_served=total_predictions,
    )
