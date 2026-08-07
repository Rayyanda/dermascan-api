"""
FastAPI dependency for protected routes — reads the "Authorization: Bearer
<token>" header, validates the JWT, and returns the current User row.
"""

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database import get_db
from app.models_db import User
from app.services.auth_service import decode_access_token

_bearer_scheme = HTTPBearer(
    scheme_name="JWT",
    description="Paste the access_token returned by POST /api/auth/login",
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token tidak valid atau sudah kedaluwarsa.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        user_id = decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise unauthorized

    user = db.query(User).get(user_id)
    if user is None:
        raise unauthorized
    return user
