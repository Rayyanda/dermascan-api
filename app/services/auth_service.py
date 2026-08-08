"""
Auth Service — password hashing (bcrypt) and plain username/password
register+login for Flutter app end users. No token: the Flutter app just
holds onto the numeric user id after login/register (see app/config.py
for the security trade-off this implies).
"""

import datetime as dt
from typing import Optional

import bcrypt
from sqlalchemy.orm import Session

from app.models_db import User


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def register_user(
    db: Session,
    username: str,
    password: str,
    full_name: str,
    birth_date: Optional[str] = None,
    gender: Optional[str] = None,
    weight: Optional[float] = None,
    height: Optional[float] = None,
    medical_history: Optional[str] = None,
    family_history: Optional[str] = None,
    outdoor_activity: Optional[str] = None,
) -> User:
    existing = db.query(User).filter(User.username == username).first()
    if existing is not None:
        raise ValueError("Username sudah digunakan.")

    user = User(
        username=username,
        password_hash=hash_password(password),
        full_name=full_name,
        birth_date=birth_date,
        gender=gender,
        weight=weight,
        height=height,
        medical_history=medical_history,
        family_history=family_history,
        outdoor_activity=outdoor_activity,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None

    user.last_login = dt.datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user


def compute_age(birth_date: Optional[str]) -> Optional[int]:
    """
    Best-effort age calculation. birth_date is stored as whatever raw
    string the Flutter app sent, so this tries a few common formats
    before giving up and returning None (rather than raising) — a
    profile should still load even if age can't be computed.
    """
    if not birth_date:
        return None

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            born = dt.datetime.strptime(birth_date.strip(), fmt).date()
            today = dt.date.today()
            age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
            return age
        except ValueError:
            continue
    return None


def compute_bmi(weight: Optional[float], height: Optional[float]) -> Optional[float]:
    """weight in kg, height in cm (standard convention)."""
    if not weight or not height:
        return None
    try:
        height_m = height / 100
        return round(weight / (height_m ** 2), 2)
    except (ZeroDivisionError, TypeError):
        return None


def user_to_profile_dict(user: User) -> dict:
    """Matches Flutter's UserProfile.fromJson exactly — field names and all."""
    return {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "age": compute_age(user.birth_date),
        "gender": user.gender,
        "weight": user.weight,
        "height": user.height,
        "bmi": compute_bmi(user.weight, user.height),
        "medical_history": user.medical_history,
        "family_history": user.family_history,
        "outdoor_activity": user.outdoor_activity,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }
