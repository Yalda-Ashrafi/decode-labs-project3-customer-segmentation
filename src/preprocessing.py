"""
preprocessing.py
----------------
Phase 1 of the DecodeLabs Project 3 pipeline: SCALE.
Every function is pure (input -> output, no hidden state) so the same code
runs identically in the Streamlit app, in a notebook and in run_pipeline.py.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

GENDER_COL = "Gender"
NUMERIC_COLS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
ID_COL = "CustomerID"

# Behavioural features used for clustering by default. Gender is encoded and
# kept for PROFILING, but excluded from the distance metric by default.
DEFAULT_FEATURES = NUMERIC_COLS


def load_data(path: str) -> pd.DataFrame:
    """Read the CSV and normalise column whitespace."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column audit: dtype, missing count, % missing, uniques."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
        }
    ).reset_index(names="column")


def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values without dropping customers.
        Numeric  -> median  (robust to income/age outliers)
        Object   -> mode    (most frequent category)
    Returns the clean frame and a log of what was filled.
    """
    out = df.copy()
    log: dict[str, dict] = {}

    for col in out.columns:
        n_missing = int(out[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            fill = out[col].median()
            strategy = "median"
        else:
            fill = out[col].mode(dropna=True)
            fill = fill.iloc[0] if not fill.empty else "Unknown"
            strategy = "mode"
        out[col] = out[col].fillna(fill)
        log[col] = {"filled": n_missing, "strategy": strategy, "value": fill}

    before = len(out)
    out = out.drop_duplicates()
    if len(out) < before:
        log["__duplicates__"] = {
            "filled": before - len(out),
            "strategy": "drop_duplicates",
            "value": "-",
        }
    return out.reset_index(drop=True), log


def encode_gender(df: pd.DataFrame, col: str = GENDER_COL) -> pd.DataFrame:
    """
    Map Gender -> Gender_Encoded (Male = 0, Female = 1).
    Binary label encoding beats one-hot here: a two-level category produces
    two perfectly collinear dummies, which double-count inside PCA.
    """
    out = df.copy()
    if col not in out.columns:
        return out
    normalised = out[col].astype(str).str.strip().str.title()
    out["Gender_Encoded"] = normalised.map({"Male": 0, "Female": 1})
    if out["Gender_Encoded"].isna().any():          # typos, 'Other', blanks
        out["Gender_Encoded"] = out["Gender_Encoded"].fillna(
            out["Gender_Encoded"].mode().iloc[0]
        )
    out["Gender_Encoded"] = out["Gender_Encoded"].astype(int)
    return out


def build_feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select the modelling columns and guarantee they are numeric."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Features not found in dataframe: {missing}")
    return df[features].astype(float)


def scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    z = (x - mu) / sigma

    Without this a $137k income axis mathematically swallows a 1-100 spending
    axis and K-Means clusters on income alone. The fitted scaler is returned so
    centroids can later be inverse-transformed back into human units.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index), scaler


def run_preprocessing(path: str, features: list[str] | None = None) -> dict:
    """Convenience wrapper that executes the whole of Phase 1 in order."""
    features = features or DEFAULT_FEATURES
    raw = load_data(path)
    report = quality_report(raw)
    clean, missing_log = handle_missing(raw)
    clean = encode_gender(clean)
    X = build_feature_matrix(clean, features)
    X_scaled, scaler = scale_features(X)
    return {
        "raw": raw, "clean": clean, "quality_report": report,
        "missing_log": missing_log, "features": features,
        "X": X, "X_scaled": X_scaled, "scaler": scaler,
    }