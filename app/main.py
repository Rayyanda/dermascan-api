"""
DermaScan Backend — main FastAPI application.

Two audiences share this one backend:
  1. AI Studio (web)     -> served at "/"  (static/ dashboard) + /api/dataset,
                             /api/training, /api/models, /api/dashboard
  2. Flutter mobile app   -> talks only to /api/predict

Run with:  python run.py
Docs at:   http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import APP_TITLE, APP_VERSION, BASE_DIR
from app.database import init_db, SessionLocal
from app.services import model_service
from app.routers import dataset, training, models, prediction, dashboard, auth

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# CORS: the AI Studio web dashboard and the Flutter app (via http on a
# device/emulator) both need to reach this API from a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    # If a trained .keras file was committed to git (data/models/) but the
    # DB is fresh/ephemeral (e.g. Render free tier, no persistent disk),
    # register + auto-deploy it so /api/predict works immediately.
    db = SessionLocal()
    try:
        model_service.sync_committed_models(db)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------
app.include_router(dataset.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(prediction.router)
app.include_router(dashboard.router)
app.include_router(auth.router)


@app.get("/api/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": APP_TITLE, "version": APP_VERSION}


# ---------------------------------------------------------------------------
# AI Studio static web dashboard (served last so /api/* routes above win)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="ai-studio")
