"""Data merge, cleaning, validation and EDA (Member 4).

Merges customer_transactions.csv with customer_social_profiles.csv into one row per
transaction, checks the merge, adds the engineered features, and saves the EDA figures.

    data/raw/customer_transactions.csv  ─┐
                                         ├─> data/processed/merged_dataset.csv
    data/raw/customer_social_profiles.csv┘   data/processed/merge_report.json
                                             data/processed/eda_figures/*.png

Run (from the repo root):  python scripts/data_merge.py

Cleaning and feature choices match Member 1's notebook (notebooks/Formative2ML.ipynb), so
merged_dataset.csv is the same file whichever one produced it; this script adds the
validation checks and saves every figure to disk for the report.
"""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot: save figures, no window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import MERGED_DATASET, PROCESSED_DIR, RAW_DIR, ROOT

SOCIAL_CSV = RAW_DIR / "customer_social_profiles.csv"
TRANS_CSV = RAW_DIR / "customer_transactions.csv"
FIGURE_DIR = PROCESSED_DIR / "eda_figures"
REPORT_JSON = PROCESSED_DIR / "merge_report.json"
TARGET = "product_category"
NUM_COLS = ["purchase_amount", "customer_rating", "avg_engagement",
            "avg_purchase_interest", "n_social_profiles"]


# ---------------------------------------------------------------------------
# 1. Load and profile
# ---------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    social = pd.read_csv(SOCIAL_CSV)
    trans = pd.read_csv(TRANS_CSV)
    return social, trans


def profile(name: str, df: pd.DataFrame) -> dict:
    """Shape, types, nulls and duplicates for one table."""
    info = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nulls": {c: int(n) for c, n in df.isna().sum().items() if n},
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }
    print(f"\n--- {name}: {info['rows']} rows x {info['columns']} cols")
    print(df.dtypes.to_string())
    print(f"nulls: {info['nulls'] or 'none'}   exact duplicate rows: {info['exact_duplicate_rows']}")
    return info


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def clean(social: pd.DataFrame, trans: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    social, trans = social.copy(), trans.copy()

    # Social IDs look like "A178"; transaction IDs are the plain integer 178.
    bad_ids = social.loc[~social["customer_id_new"].astype(str).str.fullmatch(r"A\d+"), "customer_id_new"]
    assert bad_ids.empty, f"unexpected social customer IDs: {bad_ids.tolist()}"

    before = len(social)
    social = social.drop_duplicates()
    social["customer_id"] = social["customer_id_new"].str[1:].astype(int)
    report["social_exact_duplicates_dropped"] = before - len(social)

    assert trans["transaction_id"].is_unique, "transaction_id must be unique"
    trans["customer_id"] = trans["customer_id_legacy"].astype(int)
    trans["purchase_date"] = pd.to_datetime(trans["purchase_date"], format="%Y-%m-%d")

    # customer_rating: median, because it is robust to the skew of a 1-5 scale.
    missing = int(trans["customer_rating"].isna().sum())
    median_rating = float(trans["customer_rating"].median())
    trans["customer_rating"] = trans["customer_rating"].fillna(median_rating)
    report["ratings_imputed"] = {"count": missing, "value": median_rating}

    # Range checks: every score must be inside its documented scale.
    checks = {
        "purchase_amount > 0": bool((trans["purchase_amount"] > 0).all()),
        "customer_rating in [1, 5]": bool(trans["customer_rating"].between(1, 5).all()),
        "engagement_score in [0, 100]": bool(social["engagement_score"].between(0, 100).all()),
        "purchase_interest_score in [1, 5]": bool(social["purchase_interest_score"].between(1, 5).all()),
        "review_sentiment in {Positive, Neutral, Negative}":
            bool(social["review_sentiment"].isin(["Positive", "Neutral", "Negative"]).all()),
    }
    report["range_checks"] = checks
    assert all(checks.values()), f"range check failed: {checks}"

    print(f"\ndropped {report['social_exact_duplicates_dropped']} exact duplicate social rows "
          f"({before} -> {len(social)}); imputed {missing} missing ratings with median {median_rating}")
    return social, trans


# ---------------------------------------------------------------------------
# 3. Merge
# ---------------------------------------------------------------------------
def mode_first(s: pd.Series):
    """Most frequent value; ties go to the alphabetically first so the result is reproducible."""
    return s.mode().iloc[0]


def aggregate_social(social: pd.DataFrame) -> pd.DataFrame:
    """One row per customer.

    A customer can have several social profiles (different platforms, and repeat rows for
    the same platform). Joining the raw rows would copy each transaction once per profile
    and inflate the data, so the profiles are summarised first.
    """
    return social.groupby("customer_id").agg(
        avg_engagement=("engagement_score", "mean"),
        avg_purchase_interest=("purchase_interest_score", "mean"),
        n_social_profiles=("social_media_platform", "count"),
        top_platform=("social_media_platform", mode_first),
        top_sentiment=("review_sentiment", mode_first),
    ).reset_index()


def merge(trans: pd.DataFrame, social_agg: pd.DataFrame, report: dict) -> pd.DataFrame:
    """LEFT join on customer_id: every transaction is kept because the target
    (product_category) lives in the transactions; a social profile without a
    purchase has no label and cannot be used for training."""
    t_ids, s_ids = set(trans["customer_id"]), set(social_agg["customer_id"])
    merged = trans.merge(social_agg, on="customer_id", how="left", validate="m:1", indicator=True)

    report["keys"] = {
        "transaction_customers": len(t_ids),
        "social_customers": len(s_ids),
        "customers_in_both": len(t_ids & s_ids),
        "transaction_customers_without_social": len(t_ids - s_ids),
        "social_customers_without_transactions": len(s_ids - t_ids),
    }
    report["join"] = {"type": "left on customer_id (many transactions : one social summary)",
                      **{k: int(v) for k, v in merged["_merge"].value_counts().items()}}

    # Customers with no social profile: medians for scores, 0 profiles, "Unknown" categories.
    num = ["avg_engagement", "avg_purchase_interest"]
    report["post_merge_nulls_before_fill"] = {c: int(n) for c, n in merged.isna().sum().items() if n}
    merged[num] = merged[num].fillna(merged[num].median())
    merged["n_social_profiles"] = merged["n_social_profiles"].fillna(0)
    merged[["top_platform", "top_sentiment"]] = merged[["top_platform", "top_sentiment"]].fillna("Unknown")
    return merged.drop(columns="_merge")


def validate_merge(trans: pd.DataFrame, social: pd.DataFrame, merged: pd.DataFrame, report: dict) -> None:
    """Post-merge checks. Any failure stops the script."""
    matched_ids = sorted(set(trans["customer_id"]) & set(social["customer_id"]))
    spot_id = matched_ids[0]
    raw = social[social["customer_id"] == spot_id]
    row = merged[merged["customer_id"] == spot_id].iloc[0]

    checks = {
        "row count unchanged (no rows gained or lost)": len(merged) == len(trans),
        "transaction_id still unique": bool(merged["transaction_id"].is_unique),
        "purchase_amount total unchanged": int(merged["purchase_amount"].sum()) == int(trans["purchase_amount"].sum()),
        "product_category distribution unchanged":
            merged[TARGET].value_counts().equals(trans[TARGET].value_counts()),
        "no nulls left": int(merged.isna().sum().sum()) == 0,
        f"spot check customer {spot_id}: avg_engagement matches raw rows":
            np.isclose(row["avg_engagement"], raw["engagement_score"].mean()),
        f"spot check customer {spot_id}: n_social_profiles matches raw rows":
            int(row["n_social_profiles"]) == len(raw),
    }
    report["post_merge_checks"] = {k: bool(v) for k, v in checks.items()}
    print("\npost-merge checks:")
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    assert all(checks.values()), "post-merge validation failed"


# ---------------------------------------------------------------------------
# 4. Feature engineering (same as Member 1's notebook)
# ---------------------------------------------------------------------------
def engineer(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_weekday"] = df["purchase_date"].dt.dayofweek
    df["is_weekend"] = (df["purchase_weekday"] >= 5).astype(int)
    df["cust_avg_spend"] = df.groupby("customer_id")["purchase_amount"].transform("mean")
    df["cust_purchase_count"] = df.groupby("customer_id")["transaction_id"].transform("count")
    df["interest_x_engagement"] = df["avg_engagement"] * df["avg_purchase_interest"]
    df["high_value_purchase"] = (df["purchase_amount"] > df["purchase_amount"].median()).astype(int)
    return df


# ---------------------------------------------------------------------------
# 5. EDA figures
# ---------------------------------------------------------------------------
def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=120)
    plt.close(fig)


def eda(df: pd.DataFrame, report: dict) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    summary = df[NUM_COLS].describe().T.round(2)
    print("\nsummary statistics:\n" + summary.to_string())
    report["summary_statistics"] = summary.to_dict(orient="index")
    report["target_counts"] = {k: int(v) for k, v in df[TARGET].value_counts().items()}

    # 1. Distributions
    fig, axes = plt.subplots(1, len(NUM_COLS), figsize=(16, 3.6))
    for ax, c in zip(axes, NUM_COLS):
        sns.histplot(df[c], bins=20, ax=ax, color="steelblue")
        ax.set_title(c, fontsize=10)
        ax.set_xlabel(c)
        ax.set_ylabel("transactions")
    fig.suptitle("Distribution of numeric features (merged dataset)")
    save(fig, "01_distributions.png")

    # 2. Outliers (IQR rule, the same rule the box plot whiskers use)
    outliers = {}
    for c in NUM_COLS:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    report["iqr_outliers"] = outliers
    fig, axes = plt.subplots(1, len(NUM_COLS), figsize=(16, 3.6))
    for ax, c in zip(axes, NUM_COLS):
        sns.boxplot(y=df[c], ax=ax, color="lightsteelblue")
        ax.set_title(f"{c}\n({outliers[c]} IQR outliers)", fontsize=10)
    fig.suptitle("Outlier check (box plots)")
    save(fig, "02_outliers.png")

    # 3. Correlations
    corr = df[NUM_COLS + ["interest_x_engagement", "cust_avg_spend"]].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation heat-map (Pearson)")
    save(fig, "03_correlations.png")

    # 4. Target balance
    fig, ax = plt.subplots(figsize=(6.5, 4))
    counts = df[TARGET].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="steelblue")
    ax.axhline(len(df) / len(counts), ls="--", c="grey", label="perfectly balanced")
    ax.set_title("Target: product_category balance")
    ax.set_xlabel("product_category")
    ax.set_ylabel("transactions")
    ax.legend()
    save(fig, "04_target_balance.png")

    # 5. Social features vs target
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    sns.countplot(data=df, x="top_platform", hue=TARGET, ax=axes[0])
    axes[0].set_title("Product category by main social platform")
    axes[0].set_ylabel("transactions")
    axes[0].legend(title=TARGET, fontsize=8, title_fontsize=8, ncol=5,
                   loc="upper center", bbox_to_anchor=(0.5, -0.14))
    sns.boxplot(data=df, x=TARGET, y="avg_purchase_interest", ax=axes[1], color="lightsteelblue")
    axes[1].set_title("Purchase-interest score by product category")
    save(fig, "05_social_vs_target.png")

    # 6. Merge coverage
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    has_social = np.where(df["n_social_profiles"] > 0, "matched social profile", "no social profile")
    sns.countplot(x=has_social, ax=ax, color="steelblue")
    ax.set_title("Merge coverage (transactions)")
    ax.set_ylabel("transactions")
    save(fig, "06_merge_coverage.png")


# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    report: dict = {}
    social, trans = load_raw()
    print("=== Data merge + EDA ===")
    report["raw"] = {"social": profile("customer_social_profiles", social),
                     "transactions": profile("customer_transactions", trans)}

    social, trans = clean(social, trans, report)
    social_agg = aggregate_social(social)
    print(f"social rows per customer: min {social.groupby('customer_id').size().min()}, "
          f"max {social.groupby('customer_id').size().max()} -> aggregated to {len(social_agg)} customers")

    merged = merge(trans, social_agg, report)
    print(f"join: {report['join']}")
    print(f"keys: {report['keys']}")
    validate_merge(trans, social, merged, report)

    merged = engineer(merged)
    eda(merged, report)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_DATASET, index=False, date_format="%Y-%m-%d")
    report["output"] = {"file": str(MERGED_DATASET.relative_to(ROOT)), "rows": len(merged), "columns": merged.shape[1]}
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nsaved {MERGED_DATASET.relative_to(ROOT)} ({merged.shape[0]} x {merged.shape[1]}), "
          f"{REPORT_JSON.relative_to(ROOT)}, figures in {FIGURE_DIR.relative_to(ROOT)}/")
    return merged


if __name__ == "__main__":
    main()
