"""
Pydantic schemas — the JSON contracts exposed to both the AI Studio web
dashboard and the Flutter mobile app. Keep these stable; the Flutter side
should be implemented to match these shapes exactly.
"""

import datetime as dt
from typing import Optional, List, Dict

from pydantic import BaseModel, Field, EmailStr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ClassStat(BaseModel):
    class_label: str
    short_label: str
    full_label: str
    image_count: int
    
class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=1, max_length=120)


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    created_at: dt.datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class PredictionHistoryItem(BaseModel):
    prediction_id: int
    predicted_class: str
    short_label: str
    full_label: str
    malignant_potential: str
    confidence: float
    is_dummy_prediction: bool
    created_at: dt.datetime


class DatasetStatsResponse(BaseModel):
    total_images: int
    per_class: List[ClassStat]
    ready_to_train: bool
    min_images_per_class_required: int


class DatasetImageOut(BaseModel):
    id: int
    filename: str
    class_label: str
    uploaded_at: dt.datetime

    class Config:
        from_attributes = True


class DatasetUploadResponse(BaseModel):
    uploaded: int
    skipped: int
    details: List[str]


class DatasetSyncResponse(BaseModel):
    registered: int
    unknown_class_folders: int
    details: List[str]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainingStartRequest(BaseModel):
    epochs: int = Field(default=30, ge=1, le=200)
    batch_size: int = Field(default=32, ge=4, le=256)
    learning_rate: float = Field(default=1e-4, gt=0)


class TrainingRunOut(BaseModel):
    id: int
    status: str
    architecture: str
    epochs: int
    batch_size: int
    learning_rate: float
    num_train_images: int
    num_val_images: int
    train_accuracy: Optional[float]
    val_accuracy: Optional[float]
    train_loss: Optional[float]
    val_loss: Optional[float]
    error_message: Optional[str]
    started_at: Optional[dt.datetime]
    finished_at: Optional[dt.datetime]
    created_at: dt.datetime

    class Config:
        from_attributes = True


class TrainingRunDetail(TrainingRunOut):
    classification_report: Optional[Dict] = None
    confusion_matrix: Optional[List[List[int]]] = None
    history: Optional[Dict[str, List[float]]] = None


# ---------------------------------------------------------------------------
# Models (export / deployment)
# ---------------------------------------------------------------------------
class ModelVersionOut(BaseModel):
    id: int
    filename: str
    val_accuracy: Optional[float]
    is_deployed: bool
    created_at: dt.datetime
    deployed_at: Optional[dt.datetime]
    training_run_id: Optional[int]

    class Config:
        from_attributes = True


class DeployResponse(BaseModel):
    message: str
    deployed_model: ModelVersionOut


# ---------------------------------------------------------------------------
# Prediction  (the contract the Flutter app talks to)
# ---------------------------------------------------------------------------
class ClassProbability(BaseModel):
    class_label: str          # "mel"
    short_label: str          # "MEL"
    full_label: str           # "Melanoma"
    probability: float        # 0.0 - 1.0


class PredictionResponse(BaseModel):
    predicted_class: str
    short_label: str
    full_label: str
    malignant_potential: str        # "benign" | "pre-malignant" | "malignant"
    confidence: float               # top-1 probability, 0.0 - 1.0
    all_probabilities: List[ClassProbability]
    is_dummy_prediction: bool       # true if no trained model has been deployed yet
    model_version_id: Optional[int]
    disclaimer: str = (
        "DermaScan is intended for educational purposes and preliminary "
        "screening only. It is not a medical diagnostic tool."
    )
    prediction_id: int
    created_at: dt.datetime


# ---------------------------------------------------------------------------
# Dashboard (AI Studio home)
# ---------------------------------------------------------------------------
class DashboardSummary(BaseModel):
    total_dataset_images: int
    total_training_runs: int
    active_model: Optional[ModelVersionOut]
    last_training_run: Optional[TrainingRunOut]
    total_predictions_served: int
