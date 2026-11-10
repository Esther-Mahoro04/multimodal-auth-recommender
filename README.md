# Face + Voice Authenticated Product Recommender

Formative 2: Multimodal Data Preprocessing.

A user must pass a **face check** and then a **voice check** before the system
shows a **product recommendation**. If either check fails, access is denied.

## Folder structure

```
data/raw/          original datasets (customer_social_profiles.csv, customer_transactions.csv)
data/processed/    merged_dataset.csv, model_features.csv, image_features.csv, audio_features.csv
images/            face photos: <name>_neutral.jpg, <name>_smiling.jpg, <name>_surprised.jpg
audio/             voice clips: <name>_approve.wav, <name>_confirm.wav
models/            product_model.pkl, face_model.pkl, voice_model.pkl
scripts/           config.py, product_model.py, face_auth.py, voice_auth.py, main.py
notebooks/         Jupyter notebooks
```

## Team conventions (agreed so nothing breaks)

- **Names:** lowercase first names, no spaces (`alice_neutral.jpg`, `alice_confirm.wav`).
- **Model format:** saved with `joblib` as `.pkl` in `models/`.
- **Paths:** import them from `scripts/config.py`. Don't hard-code paths.
- **Function contracts** (Member 4's `main.py` depends on these):

| Function | File | Returns |
|---|---|---|
| `authenticate_face(image_path)` | `scripts/face_auth.py` | member name, or `None` if unknown |
| `authenticate_voice(audio_path)` | `scripts/voice_auth.py` | member name, or `None` if unknown |
| `predict_product(customer_id)` | `scripts/product_model.py` | product category (str) |

## Who owns what

| Member | Owns |
|---|---|
| Member 1 | product model, feature engineering, `scripts/product_model.py` |
| Member 2 | image pipeline, face model, `scripts/face_auth.py` |
| Member 3 | audio pipeline, voice model, `scripts/voice_auth.py` |
| Member 4 | data merge/EDA, CLI app `scripts/main.py`, demo video |

Only edit your own files. If you need to change `config.py` or a shared file, tell the team first.

## Git workflow (to avoid conflicts)

1. `git pull` before you start working.
2. Work on your own branch: `git checkout -b member1-product-model`.
3. Commit small and often: `git add <your files>`, `git commit -m "message"`.
4. Push your branch and open a Pull Request, or merge into `main` after a teammate checks it.
5. Never commit very large files (GitHub's limit is 100 MB).

## Setup and run

```bash
pip install -r requirements.txt
python scripts/main.py      # run from the repo root
```

## Team contributions

| Member | Contribution |
|---|---|
| Member 1 | |
| Member 2 | |
| Member 3 | |
| Member 4 | |

## Links

- Demo video: _add link_
- Report: _add link_
