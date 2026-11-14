# Face + Voice Authenticated Product Recommender

Formative 2: Multimodal Data Preprocessing

## What it does

A user must pass a face check and then a voice check. Only then does the system show a product recommendation. If any check fails, the user sees ACCESS DENIED and the reason.

How the system decides:

1. **Face check:** the face model must name a team member with confidence of at least 0.60, and the face must be close enough to that member's enrolled photos (cosine similarity of at least 0.40).
2. **Product model unlocked:** the prediction is made but stays hidden.
3. **Voice check:** the voice must score at least 0.5 as authorized, and it must belong to the same member as the face.
4. **Result:** if both checks pass, the recommendation is shown.

Requiring the face and the voice to match the same person is what makes the system multimodal and secure.

## Folders

- `data/raw/`: original datasets
- `data/processed/`: merged dataset, image features, audio features, EDA figures, evaluation results, access log
- `images/`: face photos (`<name>_neutral.jpg`, `<name>_smiling.jpg`, `<name>_surprised.jpg`)
- `images/unauthorized/`: images of people who are not on the team
- `audio/`: voice recordings (`<name>_approve.wav`, `<name>_confirm.wav`)
- `models/`: saved face, voice and product models
- `scripts/`: Python code and the command-line app
- `notebooks/`: Jupyter notebooks

## How to run

Run these from the main folder of the repo.

```bash
pip install -r requirements.txt
python scripts/main.py
```

Other ways to run:

```bash
# Full demo: one approved and three denied attempts
python scripts/main.py --demo --member member4

# One attempt with your own files
python scripts/main.py --face images/member4_smiling.jpg --voice audio/member4_approve.wav
```

Every attempt is saved in `data/processed/access_log.csv`.

The demo runs four scenarios:
- Authorized face and voice: approved, product shown
- Unauthorized face (an image with no face): denied at the face check
- Authorized face with a stranger's voice: denied at the voice check
- Authorized face with another member's voice: denied at the voice check

## Product model (Esther Mahoro)

- Merged `customer_social_profiles` (155 rows) and `customer_transactions` (150 rows) into one dataset of 150 rows.
- Cleaning: removed 5 duplicate rows, matched the customer ID formats (`A178` became `178`), filled 10 missing ratings with the median, converted dates.
- Social data has several rows per customer, so it was summarised to one row per customer before a left join, which keeps all 150 transactions.
- Built 22 features, for example `cust_avg_spend`, `interest_x_engagement` and date features.
- Trained Random Forest and Logistic Regression. Saved the Random Forest as `models/product_model.pkl`.
- Notebook: `notebooks/Formative2ML.ipynb`

## Image pipeline and face model (Eugene Koech)

- 10 photos. Members 3 and 4 have neutral, smiling and surprised. Members 1 and 2 have neutral and smiling.
- Faces are detected and aligned to 112 x 112 (YuNet).
- Each photo gets 8 augmentations (rotation, flip, grayscale, brightness, blur, noise), giving 90 images.
- 320 features per image: 128 SFace embedding, 160 texture (LBP) and 32 colour (HSV) values. Saved in `data/processed/image_features.csv`.
- Logistic Regression was chosen over Random Forest and XGBoost because it had the lowest log loss.

## Audio pipeline and voice model (Aime Shyaka)

- 10 recordings: 8 from the four members and 2 from unauthorized speakers.
- Augmentations (pitch shift, time stretch, noise) grew the data from 10 to 40 samples.
- 46 features: 20 MFCCs, spectral roll-off, spectral centroid and RMS energy. Saved in `data/processed/audio_features.csv`.
- Model: StandardScaler and Random Forest. Waveforms and spectrograms are in `notebooks/02_audio_eda.ipynb`.

## Data merge, EDA and command-line app (Mbithi Ndunge)

- Data merge with 7 validation checks (results in `data/processed/merge_report.json`), EDA notebook `notebooks/01_data_merge_eda.ipynb`.
- System evaluation of all three models with `scripts/evaluate_system.py`.
- Command-line app `scripts/main.py` with demo mode and an access log.

## Results

Each model was tested on data it had not seen:

- Face model: accuracy 1.00, macro-F1 1.00, log loss 0.12
- Voice model: accuracy 0.68, macro-F1 0.63, log loss 0.97
- Product model: accuracy 0.27, macro-F1 0.26, log loss 1.69. Random guessing gives 0.20, so the data has only a weak link to the product.

Full system test (3,600 face and voice pairs):
- Real face with the real person's voice: approved 70.8%
- Real face with another member's voice: approved 5.7%
- Real face with a stranger's voice: approved 24.7%

## Limitations

- 33 transactions (22%) have no social profile, so their social values are filled in.
- The product data has very little signal (150 rows, correlations of at most 0.19).
- Only two stranger voice recordings exist, so the voice model cannot reliably reject new voices.
- Members 1 and 2 have no surprised photo.

## Team rules

- Use lowercase first names for file names.
- Import paths from `scripts/config.py` and don't write your own.
- Each model has one function the app uses: `authenticate_face(image_path)`, `authenticate_voice(audio_path)` and `predict_product(customer_id)`.
- Work on your own branch and open a pull request.

## Links

- Demo video: https://drive.google.com/drive/folders/19Sth8cEp_i72_jxDOdO1u_48f8ymn_gc?usp=sharing
- Report: add link
