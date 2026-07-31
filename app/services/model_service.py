"""
Model Service — deployment logic for the AI Studio "Deploy Model" action.
Exactly one ModelVersion is ever marked is_deployed=True; that's the one
the Prediction API (and therefore the Flutter app) will use.
"""

import datetime as dt

from sqlalchemy.orm import Session

from app.config import MODELS_DIR
from app.models_db import ModelVersion
from app.ml.infer import clear_model_cache


def get_deployed_model(db: Session) -> ModelVersion | None:
    return db.query(ModelVersion).filter(ModelVersion.is_deployed == True).first()  # noqa: E712


def deploy_model(db: Session, model_version_id: int) -> ModelVersion:
    target = db.query(ModelVersion).get(model_version_id)
    if target is None:
        raise ValueError(f"ModelVersion {model_version_id} not found.")

    # Un-deploy whatever was deployed before (only one active model at a time).
    db.query(ModelVersion).filter(ModelVersion.is_deployed == True).update(  # noqa: E712
        {"is_deployed": False}
    )

    target.is_deployed = True
    target.deployed_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(target)

    clear_model_cache()  # force infer.py to reload the newly-deployed model
    return target


def sync_committed_models(db: Session) -> None:
    """
    Called once on app startup. If a .keras file sits in data/models/ but
    isn't in the database yet (e.g. the DB is a fresh/ephemeral SQLite —
    the normal case on Render's free tier, which has no persistent disk —
    while the model file itself was committed to git and therefore always
    ships with the deploy), register it as a ModelVersion.

    If nothing is currently deployed, the most recently modified known
    model is auto-deployed, so the Prediction API works immediately after
    a fresh deploy without anyone touching the AI Studio UI.
    """
    if not MODELS_DIR.exists():
        return

    known_paths = {row[0] for row in db.query(ModelVersion.filepath).all()}

    for f in sorted(MODELS_DIR.glob("*.keras")):
        if str(f) in known_paths:
            continue
        db.add(ModelVersion(
            filename=f.name,
            filepath=str(f),
            val_accuracy=None,   # unknown — this model shipped via git, not via a tracked training run
            is_deployed=False,
            notes="Auto-registered from a committed model file found in data/models/ on startup.",
        ))
    db.commit()

    if get_deployed_model(db) is None:
        candidates = db.query(ModelVersion).all()
        if candidates:
            # Prefer the most recently modified file on disk as the best guess.
            newest = max(candidates, key=lambda m: MODELS_DIR.joinpath(m.filename).stat().st_mtime
                         if MODELS_DIR.joinpath(m.filename).exists() else 0)
            deploy_model(db, newest.id)
