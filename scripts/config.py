"""Shared settings. Everyone imports from here instead of hard-coding paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = ROOT / "images"
AUDIO_DIR = ROOT / "audio"
MODELS_DIR = ROOT / "models"

# Saved files (use these exact names)
MERGED_DATASET = PROCESSED_DIR / "merged_dataset.csv"
MODEL_FEATURES = PROCESSED_DIR / "model_features.csv"
IMAGE_FEATURES = PROCESSED_DIR / "image_features.csv"
AUDIO_FEATURES = PROCESSED_DIR / "audio_features.csv"
PRODUCT_MODEL = MODELS_DIR / "product_model.pkl"
FACE_MODEL = MODELS_DIR / "face_model.pkl"
VOICE_MODEL = MODELS_DIR / "voice_model.pkl"

# TODO (team): use lowercase first names, matching the file names
# in images/ and audio/ (e.g. images/alice_neutral.jpg)
MEMBERS = ["member1", "member2", "member3", "member4"]

# TODO (Member 1): which customer ID in the dataset each member acts as
USER_TO_CUSTOMER_ID = {
    "member1": 151,
    "member2": 192,
    "member3": 114,
    "member4": 171,
}
