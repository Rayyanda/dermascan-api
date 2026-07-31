"""
Prediction API — the endpoint the Flutter app (DermaScan+) actually calls.
Single responsibility: receive an image, run inference, return a JSON
result. No web UI touches this router at all.
"""

import shutil
import uuid
import json
import datetime as dt

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import CLASS_INFO, UPLOADS_DIR, ALLOWED_IMAGE_EXTENSIONS
from app.schemas import PredictionResponse, ClassProbability
from app.services import model_service
from app.models_db import PredictionLog
from app.ml.infer import predict_image
from pathlib import Path

router = APIRouter(prefix="/api/predict", tags=["Prediction API (Flutter)"])


@router.post("", response_model=PredictionResponse)
def predict(image: UploadFile = File(...), db: Session = Depends(get_db)):
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
