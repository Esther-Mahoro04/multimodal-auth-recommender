"""Image data collection and processing (Member 2).

Pipeline for every facial image:
    load -> detect face (YuNet) -> align and crop to 112x112 -> augment
         -> extract features (SFace embedding + LBP texture histogram + HSV colour histogram)
         -> one row in data/processed/image_features.csv

Input photos go directly in images/ as <name>_neutral.jpg, <name>_smiling.jpg,
<name>_surprised.jpg. Files with any other name are ignored.

Run (from the repo root):  python scripts/image_processing.py
"""

from __future__ import annotations

import re
import urllib.request

import cv2
import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot: save figures to files, no window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

from config import IMAGE_FEATURES, IMAGES_DIR, MEMBERS, MODELS_DIR, PROCESSED_DIR, ROOT

EXPRESSIONS = ["neutral", "smiling", "surprised"]
# The face model is validated by holding out one expression, so a member needs at least two.
MIN_EXPRESSIONS = 2
IMAGE_SIZE = 112  # Input size SFace was trained on.
RANDOM_STATE = 42
FIGURE_DIR = PROCESSED_DIR / "image_figures"
AUGMENTED_IMAGE_DIR = PROCESSED_DIR / "augmented_images"
PRETRAINED_DIR = MODELS_DIR / "pretrained"

# Pre-trained ONNX networks from the OpenCV model zoo, downloaded on first use.
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
YUNET_URL = f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx"

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
EMB_DIM = 128
LBP_POINTS, LBP_RADIUS = 8, 1
LBP_BINS = LBP_POINTS + 2  # "uniform" LBP has P+2 possible codes.
LBP_GRID = 4
MAX_DETECT_SIDE = 640


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, FIGURE_DIR, AUGMENTED_IMAGE_DIR, PRETRAINED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _download(url: str):
    """Download `url` into models/pretrained/ unless it's already there; return the local path."""
    ensure_dirs()
    path = PRETRAINED_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        print(f"downloading {path.name} ...")
        urllib.request.urlretrieve(url, path)
    return path


_detector = _recognizer = None


def models():
    """Return (detector, recognizer), creating them on first call."""
    global _detector, _recognizer
    if _detector is None:
        if not hasattr(cv2, "FaceDetectorYN"):
            raise SystemExit("This OpenCV build has no FaceDetectorYN/FaceRecognizerSF. "
                             "Install a 4.x release: pip install 'opencv-python>=4.8,<5'")
        _detector = cv2.FaceDetectorYN.create(str(_download(YUNET_URL)), "", (320, 320), 0.6)
        _recognizer = cv2.FaceRecognizerSF.create(str(_download(SFACE_URL)), "")
    return _detector, _recognizer


def find_file(folder, stem):
    """Return folder/stem.<ext> for the first image extension that exists."""
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no image named {stem}{IMAGE_EXTS} in {folder}")


def enrolled_members() -> dict[str, list[str]]:
    """{name: expressions found} for everyone with enough photos in images/, e.g. alice_neutral.jpg -> "alice"."""
    pattern = re.compile(rf"(.+)_({'|'.join(EXPRESSIONS)})\.(jpg|jpeg|png)", re.IGNORECASE)
    found: dict[str, set] = {}
    for p in sorted(IMAGES_DIR.iterdir()):
        m = pattern.fullmatch(p.name)
        if m:
            found.setdefault(m.group(1), set()).add(m.group(2).lower())
    members = {}
    for name, exprs in found.items():
        missing = [e for e in EXPRESSIONS if e not in exprs]
        if len(exprs) < MIN_EXPRESSIONS:
            print(f"  WARNING: skipping {name}: missing {', '.join(missing)} photo(s)")
            continue
        if missing:
            print(f"  WARNING: {name} has no {', '.join(missing)} photo yet; the brief requires all three")
        if name not in MEMBERS:
            print(f"  NOTE: '{name}' has photos but is not listed in MEMBERS in scripts/config.py")
        members[name] = [e for e in EXPRESSIONS if e in exprs]
    return members


def load_image(path) -> np.ndarray:
    """Read an image as a BGR uint8 array, shrunk so its longest side is at most MAX_DETECT_SIDE."""
    img = cv2.imread(str(path))
    if img is None:  # imread returns None instead of raising on unreadable files.
        raise ValueError(f"could not read image {path}")
    scale = MAX_DETECT_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def detect(img: np.ndarray):
    """Return the highest-scoring face as a 15-number row (box, 5 landmarks, score) or None."""
    detector, _ = models()
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None:
        return None
    return faces[np.argmax(faces[:, -1])]


def preprocess(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Detect and align the face to 112x112. Returns (face, detected?)."""
    face = detect(img)
    if face is not None:
        _, recognizer = models()
        return recognizer.alignCrop(img, face), True
    # No face found: fall back to a centre square crop.
    h, w = img.shape[:2]
    s = min(h, w)
    crop = img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), False


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around the centre by `angle` degrees (positive = counter-clockwise)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment(img: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return {augmentation_name: image}, including the original."""
    gray3 = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    noise = rng.normal(0, 12, img.shape)
    return {
        "original":   img,
        "rotate_+15": rotate(img, 15),
        "rotate_-15": rotate(img, -15),
        "flip":       cv2.flip(img, 1),
        "grayscale":  gray3,
        "bright":     cv2.convertScaleAbs(img, alpha=1.0, beta=45),
        "dark":       cv2.convertScaleAbs(img, alpha=0.7, beta=-20),
        "blur":       cv2.GaussianBlur(img, (5, 5), 0),
        "noise":      np.clip(img + noise, 0, 255).astype(np.uint8),
    }


def embedding(face: np.ndarray) -> np.ndarray:
    """128-d SFace embedding, scaled to length 1 so cosine similarity is a dot product."""
    _, recognizer = models()
    e = recognizer.feature(face).ravel()
    return e / (np.linalg.norm(e) + 1e-9)


def lbp_histogram(face: np.ndarray) -> np.ndarray:
    """Local Binary Patterns: one normalised histogram per grid cell (16 x 10 = 160)."""
    gray = cv2.equalizeHist(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    step = IMAGE_SIZE // LBP_GRID
    hists = []
    for i in range(LBP_GRID):
        for j in range(LBP_GRID):
            cell = lbp[i * step:(i + 1) * step, j * step:(j + 1) * step]
            h, _ = np.histogram(cell, bins=LBP_BINS, range=(0, LBP_BINS))
            hists.append(h / h.sum())
    return np.concatenate(hists)


def colour_histogram(face: np.ndarray) -> np.ndarray:
    """HSV histograms: 16 hue + 8 saturation + 8 value bins (32), each summing to 1."""
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    parts = [cv2.calcHist([hsv], [0], None, [16], [0, 180]),  # OpenCV hue range is 0..179.
             cv2.calcHist([hsv], [1], None, [8], [0, 256]),
             cv2.calcHist([hsv], [2], None, [8], [0, 256])]
    return np.concatenate([p.ravel() / p.sum() for p in parts])


def extract_features(face: np.ndarray) -> np.ndarray:
    """One aligned 112x112 face -> 320 numbers = 128 embedding + 160 LBP + 32 colour."""
    return np.concatenate([embedding(face), lbp_histogram(face), colour_histogram(face)]).astype(np.float32)


def feature_names() -> list[str]:
    """Column names in the same order as extract_features()."""
    return ([f"emb_{i}" for i in range(EMB_DIM)] +
            [f"lbp_{i}" for i in range(LBP_GRID * LBP_GRID * LBP_BINS)] +
            [f"hue_{i}" for i in range(16)] + [f"sat_{i}" for i in range(8)] + [f"val_{i}" for i in range(8)])


def features_from_path(path) -> tuple[np.ndarray, bool]:
    """For inference: file -> detect/align -> features (no augmentation)."""
    face, found = preprocess(load_image(path))
    return extract_features(face), found


def show_samples(images: dict) -> None:
    """Grid: one row per member, one column per expression."""
    fig, axes = plt.subplots(len(images), len(EXPRESSIONS), figsize=(3 * len(EXPRESSIONS), 3 * len(images)), squeeze=False)
    for r, member in enumerate(images):
        for c, expr in enumerate(EXPRESSIONS):
            axes[r, c].axis("off")
            if expr in images[member]:
                axes[r, c].imshow(cv2.cvtColor(images[member][expr], cv2.COLOR_BGR2RGB))
                axes[r, c].set_title(f"{member} — {expr}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "image_samples.png", dpi=110)
    plt.close(fig)


def show_detection(img: np.ndarray, face: np.ndarray, title: str) -> None:
    """Original with detected box and landmarks, the aligned crop, and its LBP texture map."""
    vis = img.copy()
    det = detect(img)
    if det is not None:
        x, y, w, h = det[:4].astype(int)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for k in range(5):
            cv2.circle(vis, (int(det[4 + 2 * k]), int(det[5 + 2 * k])), 3, (0, 0, 255), -1)
    lbp = local_binary_pattern(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), LBP_POINTS, LBP_RADIUS, method="uniform")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, im, t in zip(axes, [cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), cv2.cvtColor(face, cv2.COLOR_BGR2RGB), lbp],
                         ["detection + landmarks", "aligned 112x112", "LBP texture codes"]):
        ax.imshow(im, cmap="gray")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "image_detection_alignment.png", dpi=110)
    plt.close(fig)


def show_augmentations(augs: dict, title: str) -> None:
    """One row showing every augmentation of a single face."""
    fig, axes = plt.subplots(1, len(augs), figsize=(2.1 * len(augs), 2.6))
    for ax, (name, im) in zip(axes, augs.items()):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "image_augmentations.png", dpi=110)
    plt.close(fig)


def show_similarity(df: pd.DataFrame) -> None:
    """Heat-map of cosine similarity between the original photos."""
    orig = df[df["augmentation"] == "original"]
    E = orig[[f"emb_{i}" for i in range(EMB_DIM)]].to_numpy()
    labels = (orig["member"] + "/" + orig["expression"].str[:4]).tolist()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(E @ E.T, vmin=-0.2, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("SFace embedding similarity (original photos)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "image_embedding_similarity.png", dpi=110)
    plt.close(fig)


def build_image_features() -> pd.DataFrame:
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_STATE)
    names = feature_names()
    rows, raw_images = [], {}
    members = enrolled_members()
    if not members:
        raise SystemExit(f"No photos found. Add <name>_neutral.jpg, <name>_smiling.jpg and <name>_surprised.jpg "
                         f"to {IMAGES_DIR.relative_to(ROOT)}/ and run again.")
    first = next(iter(members))
    example = "smiling" if "smiling" in members[first] else members[first][0]
    for member, exprs in members.items():
        raw_images[member] = {}
        out_dir = AUGMENTED_IMAGE_DIR / member
        out_dir.mkdir(parents=True, exist_ok=True)
        for expr in exprs:
            path = find_file(IMAGES_DIR, f"{member}_{expr}")
            img = load_image(path)
            raw_images[member][expr] = img
            # Align once, then augment the aligned face, so every version is a 112x112 face.
            face, found = preprocess(img)
            if not found:
                print(f"  WARNING: no face detected in {path.relative_to(ROOT)} — using a centre crop. Retake this photo.")
            augs = augment(face, rng)
            for aug_name, aug_img in augs.items():
                cv2.imwrite(str(out_dir / f"{expr}_{aug_name}.jpg"), aug_img)
                rows.append({"member": member, "expression": expr, "augmentation": aug_name,
                             "source_file": str(path.relative_to(ROOT)),
                             **dict(zip(names, extract_features(aug_img)))})
            if member == first and expr == example:
                show_detection(img, face, f"{member} / {expr}")
                show_augmentations(augs, f"Image augmentations — {member} / {expr}")
    df = pd.DataFrame(rows)
    show_samples(raw_images)
    show_similarity(df)
    df.to_csv(IMAGE_FEATURES, index=False)
    print(f"{IMAGE_FEATURES.relative_to(ROOT)}: {df.shape[0]} rows x {len(names)} features "
          f"({len(members)} members, {len(df) // df['augmentation'].nunique()} photos x "
          f"{df['augmentation'].nunique()} versions)")
    print(f"figures: {FIGURE_DIR.relative_to(ROOT)}/   augmented images: {AUGMENTED_IMAGE_DIR.relative_to(ROOT)}/")
    return df


if __name__ == "__main__":
    build_image_features()
