"""
Auth endpoints for the Flutter app (DermaScan+). Simple email/password +
JWT — no email verification, matching current project scope.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import UserRegisterRequest, UserLoginRequest, TokenResponse, UserOut
from app.services import auth_service
from app.dependencies import get_current_user
from app.models_db import User

router = APIRouter(prefix="/api/auth", tags=["Auth (Flutter)"])


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: UserRegisterRequest, db: Session = Depends(get_db)):
    try:
        user = auth_service.register_user(db, payload.email, payload.password, payload.full_name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    token = auth_service.create_access_token(user.id)
    return TokenResponse(access_token=token, user=user)


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLoginRequest, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Email atau password salah.")

    token = auth_service.create_access_token(user.id)
    return TokenResponse(access_token=token, user=user)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user
