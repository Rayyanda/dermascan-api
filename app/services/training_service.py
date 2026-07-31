"""
Training Service — creates a TrainingRun row and kicks off the actual
Keras training in a background thread, so the "Start Training" HTTP
request returns immediately (per blueprint: retraining is manual, and
the AI Studio dashboard polls for progress).
"""

import threading

from sqlalchemy.orm import Session

from app.models_db import TrainingRun, TrainingStatus
from app.ml.train import run_training


def start_training(db: Session, epochs: int, batch_size: int, learning_rate: float) -> TrainingRun:
    run = TrainingRun(
        status=TrainingStatus.PENDING,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    thread = threading.Thread(target=run_training, args=(run.id,), daemon=True)
    thread.start()

    return run
