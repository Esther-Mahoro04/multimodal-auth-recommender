"""Command-line app: face -> product model unlocked -> voice -> recommendation (Member 4).

Flow (matches the assignment diagram):
    1. Face image   -> facial recognition model. Unknown face  -> ACCESS DENIED.
    2. Face OK      -> the product model may be called; the prediction is computed but hidden.
    3. Voice sample -> voiceprint model. Unknown voice, or a voice that belongs to a
                       different member than the face             -> ACCESS DENIED.
    4. Both OK      -> the recommendation is shown to the verified user.

Run from the repo root:
    python scripts/main.py                                   # interactive
    python scripts/main.py --face images/member1_smiling.jpg --voice audio/member1_approve.wav
    python scripts/main.py --demo                            # authorized + unauthorized scenarios
    python scripts/main.py --demo --pause                    # waits for Enter between steps (for the video)
    python scripts/main.py --demo --member member2           # choose who the approved run is for

Every attempt is appended to data/processed/access_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import shlex
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

from config import AUDIO_DIR, IMAGES_DIR, MEMBERS, PROCESSED_DIR, ROOT, USER_TO_CUSTOMER_ID

# Models pickled with a slightly different scikit-learn version still load; the warning
# would only clutter the demo output.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

ACCESS_LOG = PROCESSED_DIR / "access_log.csv"
UNAUTHORIZED_IMAGES = IMAGES_DIR / "unauthorized"  # stranger photos; kept out of images/ so they're never enrolled
LINE = "─" * 64
PAUSE = False


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def ok(msg):      print(_c("  ✔ " + msg, "32"))
def fail(msg):    print(_c("  ✘ " + msg, "31"))
def info(msg):    print("    " + msg)
def header(msg):  print("\n" + _c(msg, "1"))


def step(n, title):
    if PAUSE:
        input(_c(f"\n  [Enter] step {n}: {title} ", "2"))
    else:
        time.sleep(0.15)
    print(_c(f"\n[{n}/3] {title}", "36"))


def rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(path_text: str) -> Path:
    """Accept paths relative to the current folder or the repo root; strip quotes/escapes
    that terminals add when a file is dragged in."""
    text = str(path_text).strip()
    parts = shlex.split(text) if text else []
    text = parts[0] if len(parts) == 1 else text.strip("'\"")
    p = Path(text).expanduser()
    if p.exists():
        return p
    if (ROOT / p).exists():
        return ROOT / p
    raise FileNotFoundError(f"file not found: {path_text}")


# ---------------------------------------------------------------------------
# Model calls (imported lazily so a missing package gives a clear message)
# ---------------------------------------------------------------------------
def check_face(image_path: Path) -> dict:
    """{'user': name or None, 'detail': str}"""
    import face_auth
    if hasattr(face_auth, "verify_face"):
        r = face_auth.verify_face(str(image_path))
        detail = (f"best match={r['member']}  confidence={r['confidence']:.2f}  "
                  f"cosine={r['similarity']:.2f}  -> {r['reason']}")
        return {"user": r["member"] if r["accepted"] else None, "detail": detail}
    user = face_auth.authenticate_face(str(image_path))
    return {"user": user, "detail": f"model returned {user}"}


def check_voice(audio_path: Path) -> dict:
    """{'user': name or None, 'detail': str}"""
    import voice_auth
    detail = ""
    if hasattr(voice_auth, "verify_voice"):
        _, prob = voice_auth.verify_voice(str(audio_path))
        detail = f"P(authorized voice)={prob:.2f}  "
    user = voice_auth.authenticate_voice(str(audio_path))
    return {"user": user, "detail": detail + f"speaker={user}"}


def recommend(user: str) -> tuple[int, str]:
    from product_model import predict_product
    customer_id = USER_TO_CUSTOMER_ID[user]
    return customer_id, str(predict_product(customer_id))


# ---------------------------------------------------------------------------
# One transaction
# ---------------------------------------------------------------------------
def log_attempt(row: dict) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not ACCESS_LOG.exists()
    with ACCESS_LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "face_file", "face_user", "voice_file",
                                          "voice_user", "decision", "reason", "product"])
        if new:
            w.writeheader()
        w.writerow(row)


def run_transaction(face_path, voice_path=None, ask_voice=None) -> dict:
    """Run the full flow. Give `voice_path`, or `ask_voice` (a function that returns a path)
    so the voice is only requested after the face passes."""
    log = {"timestamp": datetime.now().isoformat(timespec="seconds"), "face_file": "", "face_user": "",
           "voice_file": "", "voice_user": "", "decision": "DENIED", "reason": "", "product": ""}

    def deny(reason):
        fail(f"ACCESS DENIED — {reason}")
        log["reason"] = reason
        log_attempt(log)
        return log

    # 1. Face
    step(1, "Facial recognition")
    try:
        face_file = resolve(str(face_path))
    except FileNotFoundError as e:
        return deny(str(e))
    log["face_file"] = rel(face_file)
    info(f"image: {rel(face_file)}")
    face = check_face(face_file)
    info(face["detail"])
    if face["user"] is None:
        return deny("face not recognised")
    user = face["user"]
    log["face_user"] = user
    ok(f"face recognised: {user}")

    # 2. Product model unlocked (prediction kept hidden until the voice confirms it)
    step(2, "Product model unlocked")
    if user not in USER_TO_CUSTOMER_ID:
        return deny(f"{user} has no customer ID in scripts/config.py")
    customer_id, product = recommend(user)
    ok(f"prediction computed for customer {customer_id} (hidden until voice approval)")

    # 3. Voice
    step(3, "Voiceprint verification")
    try:
        voice_file = resolve(voice_path if voice_path is not None else ask_voice())
    except FileNotFoundError as e:
        return deny(str(e))
    log["voice_file"] = rel(voice_file)
    info(f"audio: {rel(voice_file)}")
    voice = check_voice(voice_file)
    info(voice["detail"])
    log["voice_user"] = voice["user"] or ""
    if voice["user"] is None:
        return deny("voice not approved")
    if voice["user"] != user:
        return deny(f"voice belongs to {voice['user']}, but the face belongs to {user}")
    ok(f"voice verified: {user}")

    # 4. Result
    log.update(decision="APPROVED", reason="face and voice match", product=product)
    log_attempt(log)
    print(_c(f"\n  ★ TRANSACTION APPROVED — recommended product for {user} "
             f"(customer {customer_id}): {product}", "1;32"))
    return log


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def first_existing(folder: Path, stems, exts) -> Path | None:
    for stem in stems:
        for ext in exts:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def face_of(member):
    return first_existing(IMAGES_DIR, [f"{member}_smiling", f"{member}_neutral", f"{member}_surprised"],
                          (".jpg", ".jpeg", ".png"))


def voice_of(member):
    return first_existing(AUDIO_DIR, [f"{member}_approve", f"{member}_confirm"], (".wav",))


def demo_scenarios(member: str | None = None) -> list[tuple[str, Path | None, Path | None]]:
    """Build the scenarios from the files that exist. `member` = who presents the approved run."""
    ready = [m for m in MEMBERS if face_of(m) and voice_of(m)]
    if not ready:
        raise SystemExit("No member has both a face image and a voice clip yet.")
    if member and member not in ready:
        raise SystemExit(f"{member} needs a face image in images/ and a voice clip in audio/ (have: {', '.join(ready)})")
    a = member or ("member4" if "member4" in ready else ready[0])  # member4 presents the demo video
    b = next((m for m in ready if m != a), None)

    strangers = sorted(p for p in UNAUTHORIZED_IMAGES.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) \
        if UNAUTHORIZED_IMAGES.exists() else []
    stranger_voice = first_existing(AUDIO_DIR, ["unauthorized1_approve", "unauthorized2_confirm"], (".wav",))

    scenarios = [(f"Authorized transaction ({a})", face_of(a), voice_of(a))]
    for img in strangers:
        scenarios.append((f"Unauthorized face ({img.name})", img, voice_of(a)))
    else:
        print(_c(f"  note: no stranger photo in {rel(UNAUTHORIZED_IMAGES)}/ — skipping the unauthorized-face scenario", "33"))
    if stranger_voice:
        scenarios.append((f"Valid face ({a}) + unauthorized voice", face_of(a), stranger_voice))
    if b:
        scenarios.append((f"Valid face ({a}) + another member's voice ({b})", face_of(a), voice_of(b)))
    return scenarios


def run_demo(member: str | None = None) -> int:
    header("=== Face + Voice Authenticated Product Recommender — DEMO ===")
    results = []
    for i, (title, face, voice) in enumerate(demo_scenarios(member), 1):
        header(f"{LINE}\nScenario {i}: {title}\n{LINE}")
        r = run_transaction(face, voice)
        results.append((title, r))

    header(f"{LINE}\nSummary\n{LINE}")
    for title, r in results:
        outcome = f"APPROVED → {r['product']}" if r["decision"] == "APPROVED" else f"DENIED ({r['reason']})"
        print(f"  {title:58s} {outcome}")
    print(f"\n  attempts logged to {rel(ACCESS_LOG)}")
    return 0


def run_interactive() -> int:
    header("=== Face + Voice Authenticated Product Recommender ===")
    print("  Paths can be relative to the repo root (e.g. images/member1_smiling.jpg). Type q to quit.")
    while True:
        face = input("\nPath to face image: ").strip()
        if face.lower() in ("q", "quit", "exit"):
            return 0
        run_transaction(face, ask_voice=lambda: input("Path to voice sample (\"Yes, approve\" / "
                                                      "\"Confirm transaction\"): ").strip())
        if input("\nRun another transaction? [y/N] ").strip().lower() != "y":
            return 0


def main(argv=None) -> int:
    global PAUSE
    ap = argparse.ArgumentParser(description="Face + voice authenticated product recommender")
    ap.add_argument("--face", help="face image path")
    ap.add_argument("--voice", help="voice sample path")
    ap.add_argument("--demo", action="store_true", help="run authorized + unauthorized scenarios")
    ap.add_argument("--member", help="who presents the approved demo run, e.g. member2")
    ap.add_argument("--pause", action="store_true", help="wait for Enter before each step (for recording)")
    args = ap.parse_args(argv)
    PAUSE = args.pause

    try:
        if args.demo:
            return run_demo(args.member)
        if args.face:
            r = run_transaction(args.face, args.voice,
                                ask_voice=lambda: input("Path to voice sample: ").strip())
            return 0 if r["decision"] == "APPROVED" else 1
        return run_interactive()
    except ModuleNotFoundError as e:
        sys.exit(f"Missing package '{e.name}'. Run: pip install -r requirements.txt")
    except FileNotFoundError as e:  # e.g. a model .pkl that hasn't been trained yet
        sys.exit(f"ERROR: {e}")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
        return 130


if __name__ == "__main__":
    sys.exit(main())
