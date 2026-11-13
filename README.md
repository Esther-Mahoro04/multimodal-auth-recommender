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

## Product model (Member 1: Esther Mahoro)

- Merged the social profiles and transactions into one dataset of 150 rows.
- Cleaned the data: removed 5 duplicates, fixed customer IDs, filled 10 missing ratings.
- Made 6 EDA plots and built 22 features.
- Trained Random Forest and Logistic Regression.
- Result: cross-validation accuracy 0.27 (Random Forest) and 0.22 (Logistic Regression). Random guessing gives 0.20, so the data has only a weak link to the product.
- Saved model: `models/product_model.pkl`
- Notebook: `notebooks/Formative2ML.ipynb`

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
## Member 4: Data Merge, EDA, System Evaluation & CLI App (Member 4: Ndunge Mutheu Mbithi)

### 1. Data merge and cleaning (`scripts/data_merge.py`)
Merges `customer_transactions.csv` (150 rows) and `customer_social_profiles.csv` (155 rows) into
`data/processed/merged_dataset.csv` (150 rows × 19 columns).

- **Cleaning:** dropped 5 exact duplicate social rows; matched ID formats (`A178` → `178`); filled
  10 missing `customer_rating` values with the median (3.0); converted `purchase_date` to datetime.
- **Aggregation:** customers have 1–6 social rows each, so profiles are first summarised to one row
  per customer (mean engagement, mean purchase interest, number of profiles, most frequent platform
  and sentiment). Joining the raw rows would duplicate transactions.
- **Join:** LEFT join on `customer_id` with `validate="m:1"`. Every transaction is kept because the
  target (`product_category`) only exists there.

| Customers | Count | Treatment |
|---|---|---|
| In both files | 61 | joined |
| Transactions only | 14 (33 transactions) | kept; median scores, `n_social_profiles = 0`, `Unknown` category |
| Social only | 23 | dropped (no purchase = no label) |

- **Post-merge validation (all 7 pass):** row count unchanged (150 → 150), `transaction_id` unique,
  purchase total unchanged, class counts unchanged, no nulls, and 2 spot checks against raw rows.
  Saved in `data/processed/merge_report.json`.
- **Feature engineering:** `purchase_month`, `purchase_weekday`, `is_weekend`, `cust_avg_spend`,
  `cust_purchase_count`, `interest_x_engagement`, `high_value_purchase`.

### 2. Exploratory data analysis (`notebooks/01_data_merge_eda.ipynb`)
Summary statistics, variable types, and labeled plots saved in `data/processed/eda_figures/`:
distributions, outliers (IQR), correlation heat-map, target balance, social features vs target.

- Classes are balanced (Sports 23% … Books 17%), so no resampling is needed.
- No real outliers; all flagged values are valid on their scale and were kept.
- Weak signal: every feature pair has |r| ≤ 0.19. Platform is the clearest signal (LinkedIn → Electronics).

### 3. System evaluation (`scripts/evaluate_system.py`)
Re-scores all three models out-of-fold, keeping each sample and its augmented copies in the same fold.

| Model | Validation | Accuracy | Macro-F1 | Log loss |
|---|---|---|---|---|
| Facial recognition | leave-one-expression-out | 1.000 | 1.000 | 0.122 |
| Voiceprint verification | leave-one-recording-out | 0.675 | 0.632 | 0.967 |
| Product recommendation | stratified 5-fold | 0.273 | 0.264 | 1.686 |

Multimodal simulation over 3,600 face × voice pairs: genuine pairs approved 70.8%; face + another
member's voice 5.7%; face + stranger's voice 24.7%. The voice check alone accepts 87.5% of stranger
clips, so requiring the face and voice to belong to the **same member** is what makes the system secure.
Results are in `data/processed/evaluation_summary.json` and `eda_figures/07_model_confusion.png`.

**Why multimodal matters:** on its own, the voice check accepts 87.5% of stranger clips. Because the
app requires the face and the voice to belong to the **same member**, a valid face with a stranger's
voice gets through only 24.7% of the time, and a valid face with another member's voice only 5.7%.

### How the app decides
```
face check ──✘──> ACCESS DENIED
   │ ✔
product model unlocked (prediction hidden)
   │
voice check ──✘ / different member──> ACCESS DENIED
   │ ✔ same member
recommendation shown
```


### 4. Command-line app (`scripts/main.py`)
Flow: **face** → **product model unlocked** (prediction hidden) → **voice** (must be the same
member) → **recommendation shown**. Any failed step prints `ACCESS DENIED` with the reason, and
every attempt is logged to `data/processed/access_log.csv`.

```bash
python scripts/data_merge.py                              # rebuild merged dataset + EDA figures
python scripts/evaluate_system.py                         # evaluation of all 3 models
python scripts/main.py                                    # interactive
python scripts/main.py --face images/member4_smiling.jpg --voice audio/member4_approve.wav
python scripts/main.py --demo --member member4            # all scenarios + summary table
```

| Demo scenario | Face | Voice | Result |
|---|---|---|---|
| Authorized | `member4_smiling.jpg` | `member4_approve.wav` | Approved; recommends Clothing |
| Unauthorized face | `unauthorized/no_face_object.jpg` | `member4_approve.wav` | Denied at face |
| Unauthorized voice | `member4_smiling.jpg` | `unauthorized1_approve.wav` | Denied at voice |
| Another member's voice | `member4_smiling.jpg` | `member1_approve.wav` | Denied (member mismatch) |

## Team contributions

| Member | Contribution |
|---|---|
| Member 1 |Data merge, EDA, feature engineering, product model, report section |
| Member 2 | |
| Member 3 | |
| Member 4 | | Data Merge, Exploratory Data Analysis, Data Cleaning & Merge Validation, Evaluation & Multimodal Logic, System Simulation |
## Links

- Demo video: _add link_
- Report: https://docs.google.com/document/d/1mTf_ALFRFD_z3cSxa_Ch8FMWekxfsbgnXbUW6kvxs_o/edit?usp=sharing
