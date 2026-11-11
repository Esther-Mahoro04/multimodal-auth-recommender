"""
scripts/voice_auth.py
Audio Processing, Augmentation, Feature Extraction, and Voice Verification Model (Member 3).
"""

import os
import sys
import joblib
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Ensure scripts/ directory is importable whether run from repo root or scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import AUDIO_DIR, AUDIO_FEATURES, MEMBERS, MODELS_DIR, ROOT, VOICE_MODEL  # noqa: E402

# Path configurations relative to repository root
REPO_ROOT = ROOT
AUDIO_RAW_DIR = AUDIO_DIR
PROCESSED_CSV_PATH = AUDIO_FEATURES
MODEL_SAVE_PATH = MODELS_DIR / "voice_model.joblib"
SPEAKER_MODEL_PKL_PATH = VOICE_MODEL  # models/voice_model.pkl from config.py


def _resolve_audio_path(audio_file_path: str | Path) -> Path:
    """Resolve audio path, supporting fallback for unauthorized_approve.wav -> unauthorized1_approve.wav."""
    path = Path(audio_file_path)
    if path.exists():
        return path
    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative
    # Handle shorthand names like audio/unauthorized_approve.wav -> audio/unauthorized1_approve.wav
    fallback_map = {
        "unauthorized_approve.wav": "unauthorized1_approve.wav",
        "unauthorized_confirm.wav": "unauthorized2_confirm.wav",
    }
    if path.name in fallback_map:
        candidate = path.with_name(fallback_map[path.name])
        if candidate.exists():
            return candidate
        repo_candidate = REPO_ROOT / candidate
        if repo_candidate.exists():
            return repo_candidate
    return path


# =====================================================================
# 1. AUDIO AUGMENTATION
# =====================================================================
def apply_audio_augmentations(y: np.ndarray, sr: int):
    """
    Generates >=2 acoustic augmentations:
    1. Pitch shift (+2 semitones)
    2. Time stretch (speed factor 1.1x)
    3. Additive Gaussian noise
    """
    augmentations = []

    # Augmentation 1: Pitch shift (+2 semitones)
    y_pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
    augmentations.append(("pitch_shifted", y_pitch))

    # Augmentation 2: Time stretch (1.1x speed)
    y_stretch = librosa.effects.time_stretch(y=y, rate=1.1)
    augmentations.append(("time_stretched", y_stretch))

    # Augmentation 3: Noise injection (reproducible Gaussian noise)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(len(y))
    y_noisy = y + 0.005 * noise
    augmentations.append(("noise_added", y_noisy))

    return augmentations


# =====================================================================
# 2. FEATURE EXTRACTION
# =====================================================================
def extract_audio_features(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extracts statistical acoustic features across time frames:
    - 20 MFCCs (Mean & Std) -> feat_0 .. feat_39
    - Spectral Roll-off (Mean & Std) -> feat_40, feat_41
    - Spectral Centroid (Mean & Std) -> feat_42, feat_43
    - Root Mean Square Energy (Mean & Std) -> feat_44, feat_45
    Returns a 46-dimensional feature vector.
    """
    # 1. MFCCs (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    # 3. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    # 4. Energy (RMS)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    return np.hstack([
        mfcc_mean, mfcc_std,
        rolloff_mean, rolloff_std,
        centroid_mean, centroid_std,
        rms_mean, rms_std
    ])


# =====================================================================
# 3. BUILD AND SAVE audio_features.csv
# =====================================================================
def build_features_dataset(input_dir: Path = AUDIO_RAW_DIR, output_csv: Path = PROCESSED_CSV_PATH):
    records = []
    audio_files = sorted(input_dir.glob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in: {input_dir}")

    print(f"[*] Processing {len(audio_files)} audio files from {input_dir}...")

    for file_path in audio_files:
        filename = file_path.name
        speaker_id = file_path.stem.split("_")[0]

        # Binary target: 1 = Team Member, 0 = Unauthorized / Impostor
        label = 1 if "member" in speaker_id.lower() else 0

        # Load clean source recording at uniform 22,050 Hz sample rate
        y, sr = librosa.load(str(file_path), sr=22050)

        # 1. Base clean features
        base_feats = extract_audio_features(y, sr)
        records.append({
            "filename": filename,
            "speaker_id": speaker_id,
            "augmentation": "none",
            "is_authorized": label,
            **{f"feat_{i}": val for i, val in enumerate(base_feats)}
        })

        # 2. Augmented features
        for aug_name, aug_y in apply_audio_augmentations(y, sr):
            aug_feats = extract_audio_features(aug_y, sr)
            records.append({
                "filename": filename,
                "speaker_id": speaker_id,
                "augmentation": aug_name,
                "is_authorized": label,
                **{f"feat_{i}": val for i, val in enumerate(aug_feats)}
            })

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[+] Successfully saved {len(df)} feature records to: {output_csv}")
    return df


# =====================================================================
# 4. TRAIN AND EVALUATE MODEL
# =====================================================================
def train_and_save_voice_model(csv_path: Path = PROCESSED_CSV_PATH, model_path: Path = MODEL_SAVE_PATH):
    if not csv_path.exists():
        build_features_dataset(AUDIO_RAW_DIR, csv_path)

    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    X = df[feature_cols].values
    y = df["is_authorized"].values

    print(f"\n[*] Training voice verification model on {len(df)} samples...")
    print(f"    Class distribution -> Authorized: {np.sum(y == 1)}, Impostor: {np.sum(y == 0)}")

    # Stratified K-Fold to evaluate cleanly
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies, f1_binary_scores, f1_macro_scores, losses = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                class_weight="balanced"
            ))
        ])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_val)
        probs = pipe.predict_proba(X_val)

        accuracies.append(accuracy_score(y_val, preds))
        f1_binary_scores.append(f1_score(y_val, preds, zero_division=0))
        f1_macro_scores.append(f1_score(y_val, preds, average="macro", zero_division=0))
        losses.append(log_loss(y_val, probs, labels=[0, 1]))

    print("\n" + "=" * 50)
    print("      VOICEPRINT MODEL EVALUATION RESULTS         ")
    print("=" * 50)
    print(f"Accuracy        : {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"F1-Score        : {np.mean(f1_binary_scores):.4f} (+/- {np.std(f1_binary_scores):.4f})")
    print(f"F1-Score (Macro): {np.mean(f1_macro_scores):.4f} (+/- {np.std(f1_macro_scores):.4f})")
    print(f"Log-Loss        : {np.mean(losses):.4f} (+/- {np.std(losses):.4f})")
    print("=" * 50 + "\n")

    # Fit final binary verification pipeline on all available data
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    final_pipeline.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    print(f"[+] Voice verification model saved to: {model_path}")

    # Also fit speaker identification pipeline for authenticate_voice() in scripts/main.py
    speaker_labels = np.where(df["is_authorized"] == 1, df["speaker_id"], "unauthorized")
    speaker_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    speaker_pipeline.fit(X, speaker_labels)

    bundle = {
        "binary_model": final_pipeline,
        "speaker_model": speaker_pipeline,
        "feature_cols": feature_cols,
    }
    joblib.dump(bundle, SPEAKER_MODEL_PKL_PATH)
    print(f"[+] Speaker bundle saved to: {SPEAKER_MODEL_PKL_PATH}")


# =====================================================================
# 5. INFERENCE FUNCTIONS FOR main.py
# =====================================================================
def verify_voice(audio_file_path: str, threshold: float = 0.5) -> tuple[bool, float]:
    """
    Evaluates an incoming audio sample against the trained voice model.
    Returns:
        (is_authorized: bool, confidence_score: float)
    """
    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}. Run training first.")

    resolved_path = _resolve_audio_path(audio_file_path)
    model = joblib.load(MODEL_SAVE_PATH)
    y, sr = librosa.load(str(resolved_path), sr=22050)
    feats = extract_audio_features(y, sr).reshape(1, -1)

    prob_authorized = model.predict_proba(feats)[0][1]
    is_valid = bool(prob_authorized >= threshold)
    return is_valid, float(prob_authorized)


def authenticate_voice(audio_path: str, threshold: float = 0.5) -> str | None:
    """
    Take an audio file path. Return the member's name (str) or None if unknown.
    Matches the contract required by scripts/main.py.
    """
    is_valid, _ = verify_voice(audio_path, threshold=threshold)
    if not is_valid:
        return None

    resolved_path = _resolve_audio_path(audio_path)
    y, sr = librosa.load(str(resolved_path), sr=22050)
    feats = extract_audio_features(y, sr).reshape(1, -1)

    if SPEAKER_MODEL_PKL_PATH.exists():
        loaded = joblib.load(SPEAKER_MODEL_PKL_PATH)
        if isinstance(loaded, dict) and "speaker_model" in loaded:
            speaker_model = loaded["speaker_model"]
            pred_speaker = str(speaker_model.predict(feats)[0])
            if pred_speaker in MEMBERS or pred_speaker.startswith("member"):
                return pred_speaker
            return None

    # Fallback if only binary model is present
    stem_speaker = resolved_path.stem.split("_")[0]
    return stem_speaker if stem_speaker.startswith("member") else None


if __name__ == "__main__":
    # When run directly, extract features and train model
    build_features_dataset()
    train_and_save_voice_model()
