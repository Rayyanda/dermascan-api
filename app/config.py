"""
DermaScan Backend — Central Configuration
All paths, model hyperparameters, and class metadata live here so the
rest of the codebase never hardcodes a path or magic number twice.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # dermascan_backend/

# On platforms with ephemeral filesystems (Render, Railway, etc.), point
# DERMASCAN_DATA_DIR at a mounted persistent disk (e.g. Render disk mounted
# at /var/data) so the dataset, trained models, and SQLite DB survive
# restarts/redeploys. Defaults to a local "data/" folder for development.
DATA_DIR = Path(os.environ.get("DERMASCAN_DATA_DIR", str(BASE_DIR / "data")))
DATASET_DIR = DATA_DIR / "dataset"          # dataset/<class_label>/*.jpg
UPLOADS_DIR = DATA_DIR / "uploads"          # temp images sent for prediction
MODELS_DIR = DATA_DIR / "models"            # trained .keras files + metadata
DB_PATH = DATA_DIR / "dermascan.db"

for d in (DATA_DIR, DATASET_DIR, UPLOADS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# Skin lesion classes (HAM10000)
# ---------------------------------------------------------------------------
# dx_code -> metadata used by both the AI Studio dashboard and the
# Flutter app's "About Skin Cancer" / result screens.
CLASS_INFO = {
    "akiec": {
        "label": "Actinic Keratoses / Intraepithelial Carcinoma",
        "short_label": "AKIEC",
        "malignant_potential": "pre-malignant",
    },
    "bcc": {
        "label": "Basal Cell Carcinoma",
        "short_label": "BCC",
        "malignant_potential": "malignant",
    },
    "bkl": {
        "label": "Benign Keratosis-like Lesions",
        "short_label": "BKL",
        "malignant_potential": "benign",
    },
    "df": {
        "label": "Dermatofibroma",
        "short_label": "DF",
        "malignant_potential": "benign",
    },
    "mel": {
        "label": "Melanoma",
        "short_label": "MEL",
        "malignant_potential": "malignant",
    },
    "nv": {
        "label": "Melanocytic Nevi",
        "short_label": "NV",
        "malignant_potential": "benign",
    },
    "vasc": {
        "label": "Vascular Lesions",
        "short_label": "VASC",
        "malignant_potential": "benign",
    },
}

CLASS_NAMES = list(CLASS_INFO.keys())          # fixed order used by the model
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Model / training defaults (per blueprint: 00_PROJECT_BLUEPRINT.md)
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
BASE_ARCHITECTURE = "EfficientNetB0"

# Minimum images per class required before the "Start Training" action
# is allowed to run — protects against a crash on an (almost) empty dataset.
MIN_IMAGES_PER_CLASS_TO_TRAIN = 5

# ---------------------------------------------------------------------------
# Auth — Flutter app users. Deliberately simple per project scope: plain
# username/password, hashed with bcrypt. No token — the Flutter app just
# holds onto the numeric user id after login/register (query fields on
# /predict, path param on /user/{id}, etc).
#
# NOTE: this means any client that knows or guesses a user_id can read that
# user's profile/history or attribute a prediction to them — there's no
# verification tying a request to a specific logged-in session. Acceptable
# for this project's current scope; upgrade to token-based auth if this
# ever needs to hold data that matters being kept private for real.
# ---------------------------------------------------------------------------
MIN_PASSWORD_LENGTH = 6

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
APP_TITLE = "DermaScan Backend"
APP_VERSION = "0.1.0"
