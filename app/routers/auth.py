"""
Auth endpoints for the Flutter app (DermaScan+) — matches the existing
Dart client (auth_service.dart) exactly: plain username/password, no
token, response shaped as {"success": ..., "user": {...}} /
{"success": ..., "history": [...]}.

Routes are under /api (register, login, user/{id}, user/{id}/history) —
note this is a DIFFERENT prefix pattern than the prediction endpoints in
prediction.py (which are at the root, no /api prefix) — that split matches
what's already hardcoded in the Flutter app's ApiConfig usage.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import RegisterRequest, LoginRequest
from app.services import auth_service
from app.models_db import User, PredictionLog
from app.config import CLASS_INFO

router = APIRouter(prefix="/api", tags=["Auth (Flutter - simple, no token)"])


def _risk_level(class_label: str) -> str:
    return {
        "benign": "low",
        "pre-malignant": "medium",
        "malignant": "high",
    }.get(CLASS_INFO[class_label]["malignant_potential"], "low")


@router.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = auth_service.register_user(
            db,
            username=payload.username,
            password=payload.password,
            full_name=payload.full_name,
            birth_date=payload.birth_date,
            gender=payload.gender,
            weight=payload.weight,
            height=payload.height,
            medical_history=payload.medical_history,
            family_history=payload.family_history,
            outdoor_activity=payload.outdoor_activity,
        )
    except ValueError as exc:
        # Flutter checks specifically for HTTP 409 on register conflicts.
        raise HTTPException(status_code=409, detail=str(exc))

    return {"success": True, "user": auth_service.user_to_profile_dict(user)}


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, payload.username, payload.password)
    if user is None:
        # Flutter checks specifically for HTTP 401 on bad credentials.
        raise HTTPException(status_code=401, detail="Username atau password salah.")

    return {"success": True, "user": auth_service.user_to_profile_dict(user)}


@router.get("/user/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User tidak ditemukan.")

    return {"success": True, "user": auth_service.user_to_profile_dict(user)}


@router.get("/user/{user_id}/history")
def get_user_history(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User tidak ditemukan.")

    logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id == user_id)
        .order_by(PredictionLog.created_at.desc())
        .limit(200)
        .all()
    )

    history: List[dict] = [
        {
            "id": log.id,
            "predicted_class": log.predicted_class,
            "confidence": round(log.confidence, 6),
            "risk_level": _risk_level(log.predicted_class),
            "uploaded_at": log.created_at.isoformat(),
            "reviewed": log.reviewed,
            "actual_class": log.actual_class,
        }
        for log in logs
    ]
    return {"success": True, "history": history}
