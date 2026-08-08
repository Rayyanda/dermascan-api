"""
Prediction API for the Flutter app (DermaScan+) — matches the existing
Dart client (api_service.dart) exactly:
  - Routes at the ROOT level (no /api prefix): /health, /predict, /feedback
  - user_id is optional and sent as a plain multipart form field (no auth
    check that it actually belongs to whoever is asking — see the trade-off
    noted in app/config.py)
  - /predict returns HTTP 200 with {"success": false, "error": "..."} for
    handled/expected failures (bad file type, etc), reserving non-200 only
    for things Flutter explicitly branches on (500 = real server error).
"""

import shutil
import uuid
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import CLASS_INFO, UPLOADS_DIR, ALLOWED_IMAGE_EXTENSIONS
from app.schemas import FeedbackRequest
from app.services import model_service
from app.models_db import PredictionLog
from app.ml.infer import predict_image

router = APIRouter(tags=["Prediction API (Flutter)"])  # deliberately no prefix


def _risk_level(malignant_potential: str) -> str:
    return {"benign": "low", "pre-malignant": "medium", "malignant": "high"}.get(
        malignant_potential, "low"
    )


@router.get("/health")
def health(db: Session = Depends(get_db)):
    deployed = model_service.get_deployed_model(db)
    return {"status": "ok", "model_loaded": deployed is not None}


@router.post("/predict")
def predict(
    image: UploadFile = File(...),
    user_id: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
):
    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": f"Tipe file '{ext}' tidak didukung."},
        )

    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = UPLOADS_DIR / stored_name
    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    deployed = model_service.get_deployed_model(db)
    deployed_path = deployed.filepath if deployed else None

    predicted_class, prob_dict, is_dummy = predict_image(str(stored_path), deployed_path)
    class_info = CLASS_INFO[predicted_class]
    confidence = prob_dict[predicted_class]

    log = PredictionLog(
        original_filename=image.filename,
        stored_path=str(stored_path),
        predicted_class=predicted_class,
        confidence=confidence,
        all_probabilities_json=json.dumps(prob_dict),
        model_version_id=deployed.id if deployed else None,
        is_dummy_prediction=is_dummy,
        user_id=user_id,
        client="flutter",
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    all_predictions = [
        {"class": cls, "confidence": round(prob, 6)}
        for cls, prob in sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return {
        "success": True,
        "prediction": {
            "class": predicted_class,
            "confidence": round(confidence, 6),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "risk_level": _risk_level(class_info["malignant_potential"]),
        },
        "all_predictions": all_predictions,
        "disclaimer": (
            "DermaScan is intended for educational purposes and preliminary "
            "screening only. It is not a medical diagnostic tool."
        ),
        "upload_id": log.id,
    }


@router.post("/feedback")
def feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):
    log = db.query(PredictionLog).get(payload.upload_id)
    if log is None:
        return JSONResponse(status_code=404, content={"success": False, "error": "upload_id tidak ditemukan."})

    log.reviewed = True
    log.is_correct = payload.is_correct
    if payload.notes:
        log.feedback_notes = payload.notes
    db.commit()

    return {"success": True}
