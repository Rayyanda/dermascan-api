"""
Prediction API — the endpoints the Flutter app (DermaScan+) actually calls.
Requires login (Authorization: Bearer <token>) — every prediction is tied
to the authenticated user, and users can only see their own history.
"""

import shutil
import uuid
import json
import datetime as dt
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import CLASS_INFO, UPLOADS_DIR, ALLOWED_IMAGE_EXTENSIONS
from app.schemas import PredictionResponse, ClassProbability, PredictionHistoryItem
from app.services import model_service
from app.models_db import PredictionLog, User
from app.ml.infer import predict_image
from app.dependencies import get_current_user
from pathlib import Path

router = APIRouter(prefix="/api/predict", tags=["Prediction API (Flutter)"])


@router.post("", response_model=PredictionResponse)
def predict(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    # Save the incoming image temporarily (also useful as an audit trail).
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = UPLOADS_DIR / stored_name
    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    deployed = model_service.get_deployed_model(db)
    deployed_path = deployed.filepath if deployed else None

    predicted_class, prob_dict, is_dummy = predict_image(str(stored_path), deployed_path)
    class_info = CLASS_INFO[predicted_class]

    all_probs = [
        ClassProbability(
            class_label=cls,
            short_label=CLASS_INFO[cls]["short_label"],
            full_label=CLASS_INFO[cls]["label"],
            probability=round(prob, 6),
        )
        for cls, prob in sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    ]

    log = PredictionLog(
        original_filename=image.filename,
        stored_path=str(stored_path),
        predicted_class=predicted_class,
        confidence=prob_dict[predicted_class],
        all_probabilities_json=json.dumps(prob_dict),
        model_version_id=deployed.id if deployed else None,
        is_dummy_prediction=is_dummy,
        user_id=current_user.id,
        client="flutter",
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return PredictionResponse(
        predicted_class=predicted_class,
        short_label=class_info["short_label"],
        full_label=class_info["label"],
        malignant_potential=class_info["malignant_potential"],
        confidence=round(prob_dict[predicted_class], 6),
        all_probabilities=all_probs,
        is_dummy_prediction=is_dummy,
        model_version_id=deployed.id if deployed else None,
        prediction_id=log.id,
        created_at=log.created_at,
    )


@router.get("/history", response_model=List[PredictionHistoryItem])
def prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Only ever returns the logged-in user's own predictions."""
    logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id == current_user.id)
        .order_by(PredictionLog.created_at.desc())
        .limit(200)
        .all()
    )
    return [
        PredictionHistoryItem(
            prediction_id=log.id,
            predicted_class=log.predicted_class,
            short_label=CLASS_INFO[log.predicted_class]["short_label"],
            full_label=CLASS_INFO[log.predicted_class]["label"],
            malignant_potential=CLASS_INFO[log.predicted_class]["malignant_potential"],
            confidence=round(log.confidence, 6),
            is_dummy_prediction=log.is_dummy_prediction,
            created_at=log.created_at,
        )
        for log in logs
    ]