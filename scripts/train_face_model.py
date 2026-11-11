"""Facial recognition model (Member 2).

Reads data/processed/image_features.csv, compares Logistic Regression, Random Forest
and XGBoost on accuracy, macro-F1 and log loss, and saves the best to models/face_model.pkl
together with each member's enrolled embeddings, which face_auth.py uses to reject strangers.

Run (from the repo root, after image_processing.py):  python scripts/train_face_model.py
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot: save figures to files, no window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from config import FACE_MODEL, IMAGE_FEATURES, PROCESSED_DIR, ROOT
from image_processing import FIGURE_DIR, RANDOM_STATE, ensure_dirs

warnings.filterwarnings("ignore", category=UserWarning)

META_COLS = {"member", "expression", "augmentation", "source_file"}
FACE_METRICS = PROCESSED_DIR / "face_model_metrics.json"

CONFIDENCE_THRESHOLD = 0.60
# OpenCV's documented "same person" threshold for SFace is 0.363; we are slightly stricter,
# preferring to reject a member over accepting a stranger.
SIMILARITY_THRESHOLD = 0.40


def candidate_models(n_samples: int, n_features: int) -> dict:
    # PCA sits inside the pipeline so it is re-fitted on each training fold only;
    # fitting it on all the data first would leak test information.
    use_pca = n_features > n_samples
    pca = [("pca", PCA(n_components=min(30, n_samples // 2), random_state=RANDOM_STATE))] if use_pca else []
    return {
        "logistic_regression": Pipeline([("scale", StandardScaler()), *pca,
                                         ("clf", LogisticRegression(C=1.0, max_iter=5000))]),
        "random_forest": Pipeline([*pca, ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1))]),
        "xgboost": Pipeline([*pca, ("clf", XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1))]),
    }


def score(y_true, proba, n_classes: int) -> dict:
    pred = proba.argmax(axis=1)
    return {"accuracy": round(accuracy_score(y_true, pred), 4),
            "f1_macro": round(f1_score(y_true, pred, average="macro"), 4),
            "log_loss": round(log_loss(y_true, proba, labels=list(range(n_classes))), 4)}


def unit(M: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so a dot product between two rows is their cosine similarity."""
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def calibration_report(E: np.ndarray, names: np.ndarray, groups: np.ndarray) -> dict:
    """How well SIMILARITY_THRESHOLD separates members from each other.

    Each sample is matched only against the same member's samples from other expressions,
    never against its own augmented copies.
    """
    genuine, impostor = [], []
    for i in range(len(E)):
        same = (names == names[i]) & (groups != groups[i])
        other = names != names[i]
        genuine.append(float(np.max(E[same] @ E[i])))
        impostor.append(float(np.max(E[other] @ E[i])))
    genuine, impostor = np.array(genuine), np.array(impostor)
    return {"metric": "cosine", "threshold": SIMILARITY_THRESHOLD,
            "genuine_worst": round(float(genuine.min()), 4),
            "genuine_median": round(float(np.median(genuine)), 4),
            "impostor_median": round(float(np.median(impostor)), 4),
            "impostor_best": round(float(impostor.max()), 4),
            "member_FRR": round(float(np.mean(genuine < SIMILARITY_THRESHOLD)), 4),
            "member_vs_member_FAR": round(float(np.mean(impostor >= SIMILARITY_THRESHOLD)), 4)}


def train_face_model() -> dict:
    ensure_dirs()
    if not IMAGE_FEATURES.exists():
        raise SystemExit(f"{IMAGE_FEATURES.relative_to(ROOT)} not found. Run: python scripts/image_processing.py")
    df = pd.read_csv(IMAGE_FEATURES)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].to_numpy(dtype=float)
    le = LabelEncoder()
    y = le.fit_transform(df["member"])
    groups = df["expression"].to_numpy()
    n_groups, n_classes = len(set(groups)), len(le.classes_)
    if n_classes < 2:
        raise SystemExit("A classifier needs at least 2 members. Add another member's photos to images/ "
                         "and re-run image_processing.py.")
    # Leave-one-expression-out: augmented copies of a photo are near-duplicates, so a random
    # split would let the model recognise the photo instead of the person.
    cv = GroupKFold(n_splits=n_groups)

    print(f"=== Facial Recognition: {len(df)} samples, {len(feature_cols)} features, {n_classes} members, "
          f"CV = leave-one-expression-out ({n_groups} folds) ===")
    n_train = len(df) * (n_groups - 1) // n_groups
    results, oof = {}, {}
    for name, model in candidate_models(n_train, len(feature_cols)).items():
        proba = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")
        results[name] = score(y, proba, n_classes)
        oof[name] = proba
        print(f"  {name:20s} {results[name]}")

    # Highest F1; ties broken by lower log loss.
    best = max(results, key=lambda m: (results[m]["f1_macro"], -results[m]["log_loss"]))
    print(f"  -> best: {best}")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(le.inverse_transform(y), le.inverse_transform(oof[best].argmax(1)),
                                            labels=le.classes_, ax=ax, colorbar=False, xticks_rotation=45, cmap="Blues")
    ax.set_title(f"Facial recognition ({best}, out-of-fold)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "face_confusion_matrix.png", dpi=110)
    plt.close(fig)

    final = candidate_models(len(df), len(feature_cols))[best].fit(X, y)

    match_cols = [c for c in feature_cols if c.startswith("emb_")]
    E = unit(df[match_cols].to_numpy(dtype=float))
    names = df["member"].to_numpy()
    calibration = calibration_report(E, names, groups)
    print(f"  open-set matcher: {calibration}")

    bundle = {
        "model": final,
        "columns": feature_cols,
        "label_encoder": le,
        "match_columns": match_cols,
        "templates": {m: E[names == m] for m in le.classes_},
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
    FACE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, FACE_MODEL)
    FACE_METRICS.write_text(json.dumps({"best": best, "cv": results, "open_set": calibration}, indent=2))
    print(f"model saved to {FACE_MODEL.relative_to(ROOT)}, metrics to {FACE_METRICS.relative_to(ROOT)}")
    return bundle


if __name__ == "__main__":
    train_face_model()
