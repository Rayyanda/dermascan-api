"""
SQLAlchemy ORM models — the persistent state of the whole platform:
  - DatasetImage   : every image sitting in data/dataset/<class>/...
  - TrainingRun    : one row per "Start Training" click in the AI Studio
  - ModelVersion   : one row per exported model.keras artifact
  - PredictionLog  : every prediction the Flutter app has ever requested
"""

import datetime as dt
import enum

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, Enum,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from app.database import Base


class TrainingStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    birth_date = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    weight = Column(Float, nullable=True)   # kg
    height = Column(Float, nullable=True)   # cm
    medical_history = Column(Text, nullable=True)
    family_history = Column(Text, nullable=True)
    outdoor_activity = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    predictions = relationship("PredictionLog", back_populates="user")

class DatasetImage(Base):
    __tablename__ = "dataset_images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    class_label = Column(String, index=True, nullable=False)   # e.g. "mel"
    filepath = Column(String, nullable=False)                  # relative path on disk
    source = Column(String, default="manual_upload")           # manual_upload | seed_import
    uploaded_at = Column(DateTime, default=dt.datetime.utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(Enum(TrainingStatus), default=TrainingStatus.PENDING, index=True)

    architecture = Column(String, default="EfficientNetB0")
    epochs = Column(Integer, default=30)
    batch_size = Column(Integer, default=32)
    learning_rate = Column(Float, default=1e-4)

    num_train_images = Column(Integer, default=0)
    num_val_images = Column(Integer, default=0)

    train_accuracy = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)

    # Per-class metrics stored as JSON text (precision/recall/f1/support)
    classification_report_json = Column(Text, nullable=True)
    confusion_matrix_json = Column(Text, nullable=True)
    history_json = Column(Text, nullable=True)  # per-epoch acc/loss curves

    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    model_version = relationship("ModelVersion", back_populates="training_run", uselist=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True)

    filename = Column(String, nullable=False)       # e.g. model_20260705_1.keras
    filepath = Column(String, nullable=False)
    val_accuracy = Column(Float, nullable=True)

    is_deployed = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    deployed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    training_run = relationship("TrainingRun", back_populates="model_version")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, nullable=True)
    stored_path = Column(String, nullable=True)

    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    all_probabilities_json = Column(Text, nullable=False)
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    user = relationship("User", back_populates="predictions")

    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)
    is_dummy_prediction = Column(Boolean, default=False)  # true if no model was deployed yet

    client = Column(String, default="flutter")  # which client made the request
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    
    reviewed = Column(Boolean, default=False)
    is_correct = Column(Boolean, nullable=True)
    actual_class = Column(String, nullable=True)
    feedback_notes = Column(Text, nullable=True)
