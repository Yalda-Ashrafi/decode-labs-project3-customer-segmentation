"""
clustering.py
-------------
Phase 3 of the pipeline: CLUSTER.
    Gatekeeper 1 -- Elbow Method (WCSS + automatic knee detection)
    Gatekeeper 2 -- Silhouette Score (cohesion vs separation)
Davies-Bouldin and Calinski-Harabasz are added as tie-breakers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)

RANDOM_STATE = 42


def fit_kmeans(Z: pd.DataFrame, k: int) -> KMeans:
    """Fit K-Means with 10 restarts so we never land on a bad initialisation."""
    return KMeans(n_clusters=k, n_init=10, init="k-means++",
                  random_state=RANDOM_STATE).fit(Z)


def evaluate_k_range(Z: pd.DataFrame, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    """
    Sweep K and record every diagnostic in one table.
        wcss              lower is better (always falls -> needs the elbow)
        silhouette        higher is better, range [-1, 1]
        davies_bouldin    lower is better
        calinski_harabasz higher is better
    """
    rows = []
    for k in range(k_min, k_max + 1):
        km = fit_kmeans(Z, k)
        labels = km.labels_
        rows.append(
            {
                "k": k,
                "wcss": float(km.inertia_),
                "silhouette": float(silhouette_score(Z, labels)),
                "davies_bouldin": float(davies_bouldin_score(Z, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
            }
        )
    return pd.DataFrame(rows)


def find_elbow(k_values, wcss) -> int:
    """
    Kneedle-style knee detection with zero extra dependencies.
    Draw a chord from the first WCSS point to the last, then return the K
    whose perpendicular distance from that chord is largest.
    """
    k_values = np.asarray(k_values, dtype=float)
    wcss = np.asarray(wcss, dtype=float)

    # Normalise both axes to [0, 1] so WCSS units cannot dominate.
    kn = (k_values - k_values.min()) / np.ptp(k_values)
    wn = (wcss - wcss.min()) / np.ptp(wcss)

    p1, p2 = np.array([kn[0], wn[0]]), np.array([kn[-1], wn[-1]])
    unit = (p2 - p1) / np.linalg.norm(p2 - p1)

    distances = []
    for i in range(len(kn)):
        vec = np.array([kn[i], wn[i]]) - p1
        distances.append(np.linalg.norm(vec - np.dot(vec, unit) * unit))
    return int(k_values[int(np.argmax(distances))])


def recommend_k(scores: pd.DataFrame) -> dict:
    """
    Combine both gatekeepers into one defensible recommendation.
      * agree              -> that K, confidence "high"
      * adjacent           -> prefer silhouette, confidence "medium"
      * disagree by 2+     -> prefer silhouette, flagged for "review"
    Silhouette wins ties because it measures actual separation quality,
    whereas WCSS always improves as K grows.
    """
    elbow_k = find_elbow(scores["k"], scores["wcss"])
    sil_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])

    if elbow_k == sil_k:
        confidence, chosen = "high", elbow_k
    elif abs(elbow_k - sil_k) == 1:
        confidence, chosen = "medium", sil_k
    else:
        confidence, chosen = "review", sil_k

    return {
        "elbow_k": elbow_k,
        "silhouette_k": sil_k,
        "recommended_k": chosen,
        "confidence": confidence,
        "best_silhouette": float(scores["silhouette"].max()),
    }


def silhouette_breakdown(Z: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-sample silhouette values, sorted for the classic silhouette plot."""
    values = silhouette_samples(Z, labels)
    df = pd.DataFrame({"cluster": labels, "silhouette": values})
    return df.sort_values(["cluster", "silhouette"]).reset_index(drop=True)


def cluster_quality_summary(Z: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Average silhouette and size per cluster -- shows which segment is weak."""
    values = silhouette_samples(Z, labels)
    df = pd.DataFrame({"cluster": labels, "silhouette": values})
    out = df.groupby("cluster").agg(
        size=("silhouette", "size"), mean_silhouette=("silhouette", "mean")
    ).reset_index()
    out["share_pct"] = (out["size"] / out["size"].sum() * 100).round(1)
    return out