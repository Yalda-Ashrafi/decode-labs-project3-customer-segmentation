"""
dimensionality.py
-----------------
Phase 2 of the pipeline: COMPRESS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

RANDOM_STATE = 42


def fit_pca(X_scaled: pd.DataFrame, variance_threshold: float = 0.95):
    """
    Fit PCA keeping just enough components to reach the variance threshold.
    A float in (0, 1) for n_components makes sklearn solve for k such that
    sum(EVR_i) >= threshold -- the "95% Rule" slide, in code.
    """
    pca = PCA(n_components=variance_threshold, svd_solver="full",
              random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_scaled)
    cols = [f"PC{i + 1}" for i in range(pca.n_components_)]
    return pd.DataFrame(Z, columns=cols, index=X_scaled.index), pca


def variance_table(X_scaled: pd.DataFrame) -> pd.DataFrame:
    """Explained-variance curve across ALL components (scree chart data)."""
    full = PCA(svd_solver="full", random_state=RANDOM_STATE).fit(X_scaled)
    evr = full.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(evr))],
            "explained_variance": evr,
            "cumulative_variance": np.cumsum(evr),
        }
    )


def loadings_table(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """
    How much each original feature contributes to each PC.
    This is what turns "PC1 = 1.42" into "PC1 is the wealth axis".
    """
    return pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    ).round(3)


def project_for_plot(X_scaled: pd.DataFrame, n_components: int = 2):
    """
    A separate low-rank PCA used ONLY for plotting: the model may keep 3+
    components, but a scatter plot can show at most 3.
    """
    n_components = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full",
              random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_scaled)
    cols = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(Z, columns=cols, index=X_scaled.index), pca


def inverse_to_original(centroids_pca: np.ndarray, pca: PCA, scaler) -> np.ndarray:
    """
    Phase 4 translation: PCA space -> scaled space -> real-world units.
        C_original = (C_scaled * sigma) + mu
    """
    centroids_scaled = pca.inverse_transform(centroids_pca)
    return scaler.inverse_transform(centroids_scaled)