"""Product recommendation model (Member 1)."""
import joblib
import pandas as pd

from config import MODEL_FEATURES, PRODUCT_MODEL


def predict_product(customer_id):
    """Take a customer id (e.g. 151) and return the predicted product category."""
    bundle = joblib.load(PRODUCT_MODEL)
    feats = pd.read_csv(MODEL_FEATURES)
    rows = feats[feats["customer_id"] == int(customer_id)]
    if rows.empty:
        raise ValueError(f"Unknown customer id: {customer_id}")
    latest = rows.sort_values("purchase_date").iloc[[-1]]
    pred = bundle["model"].predict(latest[bundle["columns"]])
    return bundle["label_encoder"].inverse_transform(pred)[0]
