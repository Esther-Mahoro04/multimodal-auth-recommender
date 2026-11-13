"""System-level evaluation of all three models and of the multimodal decision (Member 4).

Reads only the saved feature files, so it needs no audio/image libraries:
    data/processed/image_features.csv   -> facial recognition
    data/processed/audio_features.csv   -> voiceprint verification
    data/processed/model_features.csv   -> product recommendation (+ merged_dataset.csv for the label)

Every model is scored with out-of-fold predictions, and the folds keep a sample together
with all its augmented copies (a random split would test on near-copies of training data):
    face    -> leave-one-expression-out (as in train_face_model.py)
    voice   -> leave-one-recording-out (each .wav and its 3 augmentations held out together)
    product -> stratified 5-fold

Then every out-of-fold face sample is paired with every out-of-fold voice sample and run
through the same rules as scripts/main.py, to measure what the face + voice combination adds.

Run (from the repo root):  python scripts/evaluate_system.py
Writes data/processed/evaluation_summary.json and data/processed/eda_figures/07_model_confusion.png
"""
from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import AUDIO_FEATURES, IMAGE_FEATURES, MERGED_DATASET, MODEL_FEATURES, PROCESSED_DIR, ROOT

warnings.filterwarnings("ignore")
SEED = 42
FACE_CONFIDENCE = 0.60   # same threshold as train_face_model.py
VOICE_THRESHOLD = 0.50   # same threshold as voice_auth.authenticate_voice
OUT_JSON = PROCESSED_DIR / "evaluation_summary.json"
OUT_FIG = PROCESSED_DIR / "eda_figures" / "07_model_confusion.png"


def metrics(y, proba, n_classes) -> dict:
    pred = proba.argmax(1)
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "f1_macro": round(f1_score(y, pred, average="macro"), 4),
            "log_loss": round(log_loss(y, proba, labels=list(range(n_classes))), 4)}


# ---------------------------------------------------------------- face
def evaluate_face():
    df = pd.read_csv(IMAGE_FEATURES)
    cols = [c for c in df.columns if c not in {"member", "expression", "augmentation", "source_file"}]
    le = LabelEncoder()
    y = le.fit_transform(df["member"])
    groups = df["expression"]
    n_train = len(df) * (groups.nunique() - 1) // groups.nunique()
    model = Pipeline([("scale", StandardScaler()),
                      ("pca", PCA(n_components=min(30, n_train // 2), random_state=SEED)),
                      ("clf", LogisticRegression(max_iter=5000))])
    proba = cross_val_predict(model, df[cols].to_numpy(float), y, groups=groups,
                              cv=GroupKFold(groups.nunique()), method="predict_proba")
    return df, le, y, proba, metrics(y, proba, len(le.classes_))


# ---------------------------------------------------------------- voice
def evaluate_voice():
    df = pd.read_csv(AUDIO_FEATURES)
    cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[cols].to_numpy(float)
    speaker = np.where(df["is_authorized"] == 1, df["speaker_id"], "unauthorized")
    le = LabelEncoder()
    y = le.fit_transform(speaker)
    cv = GroupKFold(df["filename"].nunique())

    def rf():
        return Pipeline([("scale", StandardScaler()), ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=SEED, class_weight="balanced"))])

    proba = cross_val_predict(rf(), X, y, groups=df["filename"], cv=cv, method="predict_proba")
    p_auth = cross_val_predict(rf(), X, df["is_authorized"], groups=df["filename"], cv=cv,
                               method="predict_proba")[:, 1]
    # Same random split as voice_auth.py, for comparison
    leaky = cross_val_predict(rf(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                              method="predict_proba")
    return df, le, y, proba, p_auth, metrics(y, proba, len(le.classes_)), metrics(y, leaky, len(le.classes_))


# ---------------------------------------------------------------- product
def evaluate_product():
    feats = pd.read_csv(MODEL_FEATURES)
    target = pd.read_csv(MERGED_DATASET)["product_category"]
    X = feats.drop(columns=["customer_id", "purchase_date"])
    le = LabelEncoder()
    y = le.fit_transform(target)
    proba = cross_val_predict(RandomForestClassifier(n_estimators=200, random_state=SEED), X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")
    m = metrics(y, proba, len(le.classes_))
    m["baseline_majority_class"] = round(float(np.bincount(y).max() / len(y)), 4)
    return le, y, proba, m


# ---------------------------------------------------------------- multimodal
def simulate_pairs(face_df, face_le, face_proba, voice_df, voice_le, voice_proba, p_auth) -> dict:
    """Pair every out-of-fold face sample with every out-of-fold voice sample and apply main.py's rules."""
    face_user = np.where(face_proba.max(1) >= FACE_CONFIDENCE, face_le.classes_[face_proba.argmax(1)], None)
    voice_pred = voice_le.classes_[voice_proba.argmax(1)]
    voice_user = np.where((p_auth >= VOICE_THRESHOLD) & (voice_pred != "unauthorized"), voice_pred, None)
    true_face = face_df["member"].to_numpy()
    true_voice = np.where(voice_df["is_authorized"] == 1, voice_df["speaker_id"], "unauthorized")

    counts = {"genuine": [0, 0], "other_member_voice": [0, 0], "stranger_voice": [0, 0]}
    for f_true, f_user in zip(true_face, face_user):
        for v_true, v_user in zip(true_voice, voice_user):
            approved = f_user is not None and v_user is not None and f_user == v_user == f_true
            kind = ("genuine" if v_true == f_true else
                    "stranger_voice" if v_true == "unauthorized" else "other_member_voice")
            counts[kind][0] += approved
            counts[kind][1] += 1

    stranger = true_voice == "unauthorized"
    return {
        "face_only_accept_rate_genuine": round(float(np.mean(face_user == true_face)), 4),
        "voice_only_stranger_accepted_as_a_member": round(float(np.mean(voice_user[stranger] != None)), 4),  # noqa: E711
        "system_true_accept_rate (genuine face + own voice)": round(counts["genuine"][0] / counts["genuine"][1], 4),
        "system_false_accept_rate (face + another member's voice)":
            round(counts["other_member_voice"][0] / counts["other_member_voice"][1], 4),
        "system_false_accept_rate (face + stranger's voice)":
            round(counts["stranger_voice"][0] / counts["stranger_voice"][1], 4),
        "pairs_tested": {k: v[1] for k, v in counts.items()},
    }


def main():
    print("=== System evaluation (out-of-fold, augmented copies never split) ===")
    face_df, face_le, face_y, face_proba, face_m = evaluate_face()
    print(f"face    (leave-one-expression-out): {face_m}")
    voice_df, voice_le, voice_y, voice_proba, p_auth, voice_m, voice_leaky = evaluate_voice()
    print(f"voice   (leave-one-recording-out):  {voice_m}")
    print(f"        (random split, as in voice_auth.py — inflated): {voice_leaky}")
    prod_le, prod_y, prod_proba, prod_m = evaluate_product()
    print(f"product (stratified 5-fold):        {prod_m}")
    system = simulate_pairs(face_df, face_le, face_proba, voice_df, voice_le, voice_proba, p_auth)
    print("multimodal decision:")
    for k, v in system.items():
        print(f"  {k}: {v}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (title, le, y, proba) in zip(axes, [
            ("Facial recognition", face_le, face_y, face_proba),
            ("Voiceprint (speaker)", voice_le, voice_y, voice_proba),
            ("Product recommendation", prod_le, prod_y, prod_proba)]):
        ConfusionMatrixDisplay.from_predictions(le.classes_[y], le.classes_[proba.argmax(1)], labels=le.classes_,
                                                ax=ax, colorbar=False, xticks_rotation=45, cmap="Blues")
        ax.set_title(f"{title}\n(out-of-fold, acc {accuracy_score(y, proba.argmax(1)):.2f})", fontsize=11)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110)
    plt.close(fig)

    OUT_JSON.write_text(json.dumps({
        "face": {"cv": "leave-one-expression-out", **face_m},
        "voice": {"cv": "leave-one-recording-out", **voice_m, "random_split_for_comparison": voice_leaky},
        "product": {"cv": "stratified 5-fold", **prod_m},
        "multimodal": system,
    }, indent=2))
    print(f"\nsaved {OUT_JSON.relative_to(ROOT)} and {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
