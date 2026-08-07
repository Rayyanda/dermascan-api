"""
Auth Service — password hashing (bcrypt) and JWT issuing/verification for
Flutter app end users. Deliberately simple, per project scope: no email
verification, no refresh tokens, no password reset flow yet.
"""

import datetime as dt

import bcrypt
import jwt
from sqlalchemy.orm import Session

from app.config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_MINUTES
from app.models_db import User


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int) -> str:
    expire = dt.datetime.utcnow() + dt.timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    """Returns the user_id encoded in the token, or raises jwt exceptions on failure."""
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    return int(payload["sub"])


def register_user(db: Session, email: str, password: str, full_name: str) -> User:
    existing = db.query(User).filter(User.email == email.lower()).first()
    if existing is not None:
        raise ValueError("Email sudah terdaftar.")

    user = User(
        email=email.lower(),
        password_hash=hash_password(password),
        full_name=full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = db.query(User).filter(User.email == email.lower()).first()
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
