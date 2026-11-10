"""Face recognition (Member 2).

Try one photo from the command line (from the repo root):
    python scripts/face_auth.py images/alice_smiling.jpg
"""
import sys

import joblib
import numpy as np

from config import FACE_MODEL, ROOT

_bundle = None


def load_bundle() -> dict:
    """Load models/face_model.pkl (saved by train_face_model.py)."""
    global _bundle
    if _bundle is None:
        if not FACE_MODEL.exists():
            raise FileNotFoundError(f"{FACE_MODEL.relative_to(ROOT)} not found. Run: "
                                    "python scripts/image_processing.py && python scripts/train_face_model.py")
        _bundle = joblib.load(FACE_MODEL)
    return _bundle


def verify_face(image_path) -> dict:
    """Full decision for one photo: best match, scores, and why it was accepted or rejected."""
    import image_processing as image

    bundle = load_bundle()
    result = {"accepted": False, "member": None, "confidence": 0.0, "similarity": 0.0, "reason": ""}

    try:
        vec, found = image.features_from_path(image_path)
    except ValueError as e:
        result["reason"] = str(e)
        return result
    if not found:
        result["reason"] = "no face detected in the image"
        return result

    row = dict(zip(image.feature_names(), vec))
    x = np.array([[row[c] for c in bundle["columns"]]], dtype=float)

    proba = bundle["model"].predict_proba(x)[0]
    idx = int(np.argmax(proba))
    name = str(bundle["label_encoder"].inverse_transform([idx])[0])
    result["member"], result["confidence"] = name, float(proba[idx])

    e = np.array([row[c] for c in bundle["match_columns"]], dtype=float)
    e = e / (np.linalg.norm(e) + 1e-9)
    result["similarity"] = float(np.max(bundle["templates"][name] @ e))

    # A classifier always names somebody, even for a stranger, so the face must also
    # match the predicted member's enrolled embeddings.
    if result["confidence"] < bundle["confidence_threshold"]:
        result["reason"] = (f"low classifier confidence "
                            f"({result['confidence']:.2f} < {bundle['confidence_threshold']:.2f})")
    elif result["similarity"] < bundle["similarity_threshold"]:
        result["reason"] = (f"does not match {name}'s enrolled photos "
                            f"(cosine {result['similarity']:.2f} < {bundle['similarity_threshold']:.2f})")
    else:
        result["accepted"], result["reason"] = True, "verified"
    return result


def authenticate_face(image_path):
    """Take an image path. Return the member's name (str) or None if unknown."""
    result = verify_face(image_path)
    return result["member"] if result["accepted"] else None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/face_auth.py <image path>")
    r = verify_face(sys.argv[1])
    print(f"{'ACCEPTED' if r['accepted'] else 'DENIED'}: best match={r['member']}  "
          f"confidence={r['confidence']:.2f}  cosine={r['similarity']:.2f}  -> {r['reason']}")
    sys.exit(0 if r["accepted"] else 1)
