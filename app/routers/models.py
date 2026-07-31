"""
Model export/deployment endpoints — AI Studio: list every trained
model version, deploy one (making it live for the Flutter Prediction
API), and download the raw .keras file.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import ModelVersionOut, DeployResponse
from app.services import model_service
from app.models_db import ModelVersion

router = APIRouter(prefix="/api/models", tags=["Model Export & Deployment"])


@router.get("", response_model=list[ModelVersionOut])
def list_models(db: Session = Depends(get_db)):
    return db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()


@router.post("/{model_id}/deploy", response_model=DeployResponse)
def deploy_model(model_id: int, db: Session = Depends(get_db)):
    try:
        model = model_service.deploy_model(db, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return DeployResponse(message=f"Model '{model.filename}' is now deployed.", deployed_model=model)


@router.get("/{model_id}/download")
def download_model(model_id: int, db: Session = Depends(get_db)):
    model = db.query(ModelVersion).get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    return FileResponse(model.filepath, filename=model.filename)
