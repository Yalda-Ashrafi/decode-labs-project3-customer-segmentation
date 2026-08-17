"""
personas.py
-----------
Phase 4 of the pipeline: TRANSLATE.
Converts abstract PCA-space centroids into business personas with a name,
tagline, demographic summary and concrete recommended actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dimensionality import inverse_to_original

INCOME_COL = "Annual Income (k$)"
SPEND_COL = "Spending Score (1-100)"
AGE_COL = "Age"


# ---------------------------------------------------------------- centroids
def centroid_table(kmeans, pca, scaler, feature_names: list[str]) -> pd.DataFrame:
    """Inverse-transform centroids from PCA space back to human units."""
    original = inverse_to_original(kmeans.cluster_centers_, pca, scaler)
    df = pd.DataFrame(original, columns=feature_names)
    df.insert(0, "cluster", range(len(df)))
    return df


def profile_clusters(clean_df: pd.DataFrame, labels: np.ndarray,
                     feature_names: list[str]) -> pd.DataFrame:
    """
    Observed profile per cluster (means of the ACTUAL members).
    The reconstructed centroid and the observed mean should match -- a cheap
    sanity check that the inverse transform is correct.
    """
    df = clean_df.copy()
    df["cluster"] = labels

    # Always profile the three business descriptors even when the user has
    # deselected one of them as a CLUSTERING feature -- the persona layer and
    # management reports need them regardless of what drove the distance metric.
    descriptors = [c for c in (AGE_COL, INCOME_COL, SPEND_COL) if c in df.columns]
    profile_cols = list(dict.fromkeys(list(feature_names) + descriptors))

    agg = {f: (f, "mean") for f in profile_cols}
    agg["size"] = ("cluster", "size")
    if "Gender_Encoded" in df.columns:
        agg["pct_female"] = ("Gender_Encoded", lambda s: s.mean() * 100)

    out = df.groupby("cluster").agg(**agg).reset_index()
    out["share_pct"] = (out["size"] / out["size"].sum() * 100).round(1)
    return out.round(2)


# ------------------------------------------------------------ level scoring
def _level(value: float, low: float, high: float) -> str:
    """Bucket a value into low / mid / high using dataset percentiles."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "mid"  # unknown -> treat as average rather than crashing
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "mid"


def _thresholds(clean_df: pd.DataFrame, col: str) -> tuple[float, float]:
    """33rd and 66th percentile of the real customer base."""
    return float(clean_df[col].quantile(0.33)), float(clean_df[col].quantile(0.66))


# -------------------------------------------------------- persona catalogue
# Keyed by (income_level, spend_level). Age shifts the tone afterwards.
PERSONA_RULES = {
    ("high", "high"): {
        "name": "High-Value Trendsetters",
        
        "tagline": "Top earners who spend freely and set the trend curve.",
        "priority": "Protect",
        "actions": [
            "Invite to an invitation-only loyalty tier with early product access.",
            "Run experiential marketing (launch events, personal styling).",
            "Assign a named account contact -- churn here is the costliest.",
        ],
    },
    ("high", "low"): {
        "name": "Affluent Conservatives",
        
        "tagline": "High income, low engagement -- the biggest untapped upside.",
        "priority": "Grow",
        "actions": [
            "Lead with quality, warranty and durability messaging, not discounts.",
            "Offer high-touch consultations to convert browsing into purchase.",
            "Test premium bundles; price is not the barrier, relevance is.",
        ],
    },
    ("low", "high"): {
        "name": "Budget Explorers",
       
        "tagline": "Modest incomes, high spending appetite and strong response rates.",
        "priority": "Nurture",
        "actions": [
            "Flash sales, bundle deals and buy-now-pay-later options.",
            "Influencer and social-first campaigns -- highest engagement per rupee.",
            "Watch credit exposure; spend already outpaces income.",
        ],
    },
    ("low", "low"): {
        "name": "Conservative Minimizers",
        
        "tagline": "Low income and low spend -- serve efficiently, not intensively.",
        "priority": "Maintain",
        "actions": [
            "Move to low-cost channels: email, app push, self-service.",
            "Promote value packs and clear entry-level price points.",
            "Cap acquisition spend; expected lifetime value is thin.",
        ],
    },
    ("mid", "mid"): {
        "name": "Mainstream Steadies",
        
        "tagline": "The dependable middle -- average income, average basket, high volume.",
        "priority": "Retain",
        "actions": [
            "Seasonal campaigns and points-based loyalty to lift frequency.",
            "Cross-sell adjacent categories from past purchases.",
            "Use as the control group for pricing and creative experiments.",
        ],
    },
}

FALLBACK_PERSONA = {
    "name": "Emerging Mid-Tier",
   
    "tagline": "A transitional segment sitting between the core groups.",
    "priority": "Observe",
    "actions": [
        "Track migration -- these customers drift into other segments over time.",
        "Test broad-appeal offers before committing budget.",
        "Re-cluster next quarter to confirm the segment is stable.",
    ],
}


def _age_prefix(age_level: str) -> str:
    return {"low": "Young ", "high": "Seasoned ", "mid": ""}[age_level]


def build_personas(clean_df: pd.DataFrame, labels: np.ndarray,
                   profile: pd.DataFrame,
                   quality: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach a persona to every cluster profile."""
    inc_lo, inc_hi = _thresholds(clean_df, INCOME_COL)
    spend_lo, spend_hi = _thresholds(clean_df, SPEND_COL)
    age_lo, age_hi = _thresholds(clean_df, AGE_COL)

    def get(row, col):
        """Safe lookup -- a descriptor may be absent on an exotic dataset."""
        return float(row[col]) if col in row.index else float("nan")

    records = []
    for _, row in profile.iterrows():
        income_level = _level(get(row, INCOME_COL), inc_lo, inc_hi)
        spend_level = _level(get(row, SPEND_COL), spend_lo, spend_hi)
        age_level = _level(get(row, AGE_COL), age_lo, age_hi)

        rule = PERSONA_RULES.get((income_level, spend_level))
        if rule is None:
            rule = PERSONA_RULES.get(("mid", "mid")) if spend_level == "mid" else None
            rule = rule or FALLBACK_PERSONA

        name = _age_prefix(age_level) + rule["name"]

        record = {
            "cluster": int(row["cluster"]),
            "persona": name,
            
            "tagline": rule["tagline"],
            "priority": rule["priority"],
            "actions": rule["actions"],
            "size": int(row["size"]),
            "share_pct": float(row["share_pct"]),
            "avg_age": round(get(row, AGE_COL), 1),
            "avg_income": round(get(row, INCOME_COL), 1),
            "avg_spend": round(get(row, SPEND_COL), 1),
            "income_level": income_level,
            "spend_level": spend_level,
            "age_level": age_level,
        }
        if "pct_female" in profile.columns:
            record["pct_female"] = round(float(row["pct_female"]), 1)
        if quality is not None:
            match = quality.loc[quality["cluster"] == row["cluster"], "mean_silhouette"]
            record["cohesion"] = round(float(match.iloc[0]), 3) if len(match) else None
        records.append(record)

    # Disambiguate identical names (possible at high K).
    out = pd.DataFrame(records)
    dupes = out["persona"].duplicated(keep=False)
    if dupes.any():
        out.loc[dupes, "persona"] = (
            out.loc[dupes, "persona"] + " #" + (out.loc[dupes].groupby("persona")
            .cumcount() + 1).astype(str)
        )
    return out


def revenue_index(personas: pd.DataFrame) -> pd.DataFrame:
    """
    Commercial ranking: size x income x spending propensity, rebased so the
    largest opportunity = 100. One number for management to prioritise with.
    """
    out = personas.copy()
    raw = out["size"] * out["avg_income"] * out["avg_spend"]
    out["opportunity_index"] = (raw / raw.max() * 100).round(1)
    return out.sort_values("opportunity_index", ascending=False).reset_index(drop=True)