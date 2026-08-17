"""
run_pipeline.py
---------------
Headless end-to-end run: SCALE -> COMPRESS -> CLUSTER -> TRANSLATE.

    python run_pipeline.py --data data/Mall_Customers.csv --variance 0.95
    python run_pipeline.py --k 5          # force a specific K
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.clustering import (cluster_quality_summary, evaluate_k_range,
                            fit_kmeans, recommend_k)
from src.dimensionality import fit_pca, loadings_table, variance_table
from src.personas import (build_personas, centroid_table, profile_clusters,
                          revenue_index)
from src.preprocessing import DEFAULT_FEATURES, run_preprocessing

OUT = Path("outputs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Customer segmentation pipeline")
    ap.add_argument("--data", default="data/Mall_Customers.csv")
    ap.add_argument("--variance", type=float, default=0.95)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--k", type=int, default=None, help="override recommended K")
    ap.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)

    # ---- Phase 1: SCALE ---------------------------------------------------
    pre = run_preprocessing(args.data, features=list(args.features))
    print(f"[1/4] Loaded {len(pre['clean'])} customers | features: {pre['features']}")
    print("      imputation log:", json.dumps(pre["missing_log"], default=str)
          if pre["missing_log"] else "      no missing values found")

    # ---- Phase 2: COMPRESS ------------------------------------------------
    Z, pca = fit_pca(pre["X_scaled"], args.variance)
    var = variance_table(pre["X_scaled"])
    print(f"[2/4] PCA kept {pca.n_components_} of {pre['X_scaled'].shape[1]} components "
          f"({pca.explained_variance_ratio_.sum():.1%} variance retained)")
    print(loadings_table(pca, pre["features"]).to_string())

    # ---- Phase 3: CLUSTER -------------------------------------------------
    scores = evaluate_k_range(Z, args.kmin, args.kmax)
    rec = recommend_k(scores)
    k = args.k or rec["recommended_k"]
    print(f"[3/4] Elbow K={rec['elbow_k']} | Silhouette K={rec['silhouette_k']} "
          f"| using K={k} (confidence: {rec['confidence']})")

    kmeans = fit_kmeans(Z, k)
    quality = cluster_quality_summary(Z, kmeans.labels_)

    # ---- Phase 4: TRANSLATE ----------------------------------------------
    profile = profile_clusters(pre["clean"], kmeans.labels_, pre["features"])
    centroids = centroid_table(kmeans, pca, pre["scaler"], pre["features"])
    personas = revenue_index(build_personas(pre["clean"], kmeans.labels_,
                                            profile, quality))
    print("[4/4] Personas:")
    print(personas[["cluster", "persona", "size", "avg_age", "avg_income",
                    "avg_spend", "opportunity_index"]].to_string(index=False))

    # ---- Persist ----------------------------------------------------------
    segmented = pre["clean"].copy()
    segmented["Cluster"] = kmeans.labels_
    segmented["Persona"] = segmented["Cluster"].map(
        dict(zip(personas["cluster"], personas["persona"])))

    segmented.to_csv(OUT / "segmented_customers.csv", index=False)
    scores.to_csv(OUT / "k_diagnostics.csv", index=False)
    var.to_csv(OUT / "explained_variance.csv", index=False)
    centroids.to_csv(OUT / "centroids_original_units.csv", index=False)
    personas.drop(columns=["actions"]).to_csv(OUT / "personas.csv", index=False)
    joblib.dump({"scaler": pre["scaler"], "pca": pca, "kmeans": kmeans,
                 "features": pre["features"]}, OUT / "model.joblib")
    print(f"\nArtifacts written to {OUT.resolve()}")


if __name__ == "__main__":
    main()