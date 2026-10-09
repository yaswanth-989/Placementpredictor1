"""
unsupervised_learning.py
========================
Unsupervised Learning module for Placement Prediction Platform:
1. Clustering: K-Means, Hierarchical (Agglomerative), and DBSCAN (Density-Based).
2. Dimensionality Reduction: PCA, t-SNE, and UMAP.
3. Anomaly Detection: Isolation Forest (with profile contrast & outlier explanations).
4. Unsupervised Synthesis & Comparative Benchmarking.
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from load_data import load_data

TARGET = "PlacementStatus"
DROP_COLS = ["StudentID", "PlacementStatus", "Salary Package", "IsAnomaly"]

# In-memory cache for fast responsive rendering
_CACHE = {}


def _get_clean_data(sample_size=2500, random_state=42):
    """
    Prepares unlabelled feature matrix and tracks raw metadata for profiling.
    Cached in memory for instant responses.
    """
    cache_key = f"data_{sample_size}_{random_state}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    df_full = load_data()
    
    # Subsample for sub-second interactive ML computation
    if len(df_full) > sample_size:
        df_sample = df_full.sample(sample_size, random_state=random_state).copy()
    else:
        df_sample = df_full.copy()

    # Retain raw columns for profiling and post-hoc validation
    raw_df = df_sample.copy()

    # Drop target and identifier columns to ensure pure unsupervised learning
    feature_df = df_sample.drop(columns=[c for c in DROP_COLS if c in df_sample.columns])

    num_cols = list(feature_df.select_dtypes(include="number").columns)
    cat_cols = list(feature_df.select_dtypes(exclude="number").columns)

    # Impute numeric features with median
    if num_cols:
        imputer = SimpleImputer(strategy="median")
        feature_df[num_cols] = imputer.fit_transform(feature_df[num_cols])

    # One-hot encode categoricals
    if cat_cols:
        encoded_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)
    else:
        encoded_df = feature_df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded_df)

    data_bundle = {
        "raw_df": raw_df,
        "feature_names": list(encoded_df.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "X_scaled": X_scaled,
        "n_samples": len(df_sample),
        "n_features": X_scaled.shape[1]
    }
    _CACHE[cache_key] = data_bundle
    return data_bundle


def _get_pca_2d():
    """Returns cached 2D PCA coordinates and fitted model for rapid scatter plotting."""
    if "pca_2d_base" in _CACHE:
        return _CACHE["pca_2d_base"]
    
    bundle = _get_clean_data()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(bundle["X_scaled"])
    
    res = {
        "pca_model": pca,
        "X_pca": X_pca,
        "exp_var": pca.explained_variance_ratio_
    }
    _CACHE["pca_2d_base"] = res
    return res


# ─────────────────────────────────────────────────────────
# 1. K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────

def run_kmeans(k=3):
    """
    Executes K-Means clustering, computes evaluation metrics,
    elbow curve (inertia & silhouette for k=2..7), and cluster persona summaries.
    """
    k = max(2, min(int(k), 8))
    cache_key = f"kmeans_k{k}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    t0 = time.time()
    bundle = _get_clean_data()
    X = bundle["X_scaled"]
    raw_df = bundle["raw_df"]
    pca_bundle = _get_pca_2d()
    X_pca = pca_bundle["X_pca"]

    # Fit requested K-Means model
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Compute validation metrics
    sil = round(float(silhouette_score(X, labels)), 4)
    db = round(float(davies_bouldin_score(X, labels)), 4)
    ch = round(float(calinski_harabasz_score(X, labels)), 2)

    # Elbow curve evaluation (k=2 to 7)
    elbow_curve = []
    for test_k in range(2, 8):
        test_km = KMeans(n_clusters=test_k, random_state=42, n_init=5)
        test_labels = test_km.fit_predict(X)
        elbow_curve.append({
            "k": test_k,
            "inertia": round(float(test_km.inertia_), 1),
            "silhouette": round(float(silhouette_score(X, test_labels)), 4)
        })

    # Cluster persona profiles
    cluster_profiles = []
    total_pts = len(labels)
    
    # Pre-calculated persona titles
    persona_names = {
        0: "High Performers (Tech & Academics)",
        1: "Core Moderate (Skill Builders)",
        2: "Academic Priority / Low Tech Prep",
        3: "Practical / Hands-on Achievers",
        4: "Emerging / Needs Mentorship",
        5: "Specialized Niche Group",
        6: "Edge Candidates",
        7: "Cluster Segment"
    }

    for c_id in range(k):
        mask = (labels == c_id)
        sub_raw = raw_df[mask]
        count = int(mask.sum())
        pct = round((count / total_pts) * 100, 1)

        # Compute averages for key metrics
        mean_cgpa = round(float(sub_raw["CGPA"].mean()), 2) if "CGPA" in sub_raw else 0
        mean_att = round(float(sub_raw["AttendancePercent"].mean()), 1) if "AttendancePercent" in sub_raw else 0
        mean_code = round(float(sub_raw["CodingTestScore"].mean()), 1) if "CodingTestScore" in sub_raw else 0
        mean_apt = round(float(sub_raw["AptitudeTestScore"].mean()), 1) if "AptitudeTestScore" in sub_raw else 0
        mean_soft = round(float(sub_raw["SoftSkillsRating"].mean()), 2) if "SoftSkillsRating" in sub_raw else 0
        mean_proj = round(float(sub_raw["Projects"].mean()), 1) if "Projects" in sub_raw else 0
        mean_intern = round(float(sub_raw["Internships"].mean()), 1) if "Internships" in sub_raw else 0
        
        # Backlog rate
        if "HistoryOfBacklogs" in sub_raw:
            backlog_rate = round(float((sub_raw["HistoryOfBacklogs"].astype(str).str.lower().isin(["yes", "1"])).mean() * 100), 1)
        else:
            backlog_rate = 0.0

        # Placement rate (for post-hoc insight)
        if "PlacementStatus" in sub_raw:
            place_rate = round(float((sub_raw["PlacementStatus"].astype(str).isin(["Placed", "1"])).mean() * 100), 1)
        else:
            place_rate = 0.0

        cluster_profiles.append({
            "cluster_id": c_id,
            "persona": persona_names.get(c_id, f"Cluster {c_id}"),
            "count": count,
            "pct": pct,
            "mean_cgpa": mean_cgpa,
            "mean_att": mean_att,
            "mean_code": mean_code,
            "mean_apt": mean_apt,
            "mean_soft": mean_soft,
            "mean_proj": mean_proj,
            "mean_intern": mean_intern,
            "backlog_rate": backlog_rate,
            "place_rate": place_rate
        })

    # Prepare scatter plot points (subsampled to 750 for super-fast client canvas rendering)
    scatter_pts = []
    step = max(1, len(labels) // 750)
    for idx in range(0, len(labels), step):
        scatter_pts.append({
            "x": round(float(X_pca[idx, 0]), 3),
            "y": round(float(X_pca[idx, 1]), 3),
            "cluster": int(labels[idx]),
            "cgpa": round(float(raw_df.iloc[idx]["CGPA"]), 2) if "CGPA" in raw_df else 0,
            "placed": str(raw_df.iloc[idx]["PlacementStatus"]) if "PlacementStatus" in raw_df else "N/A"
        })

    # Cluster centroids in 2D PCA space
    centroids_2d = []
    for c_id in range(k):
        mask = (labels == c_id)
        if mask.any():
            c_x = round(float(X_pca[mask, 0].mean()), 3)
            c_y = round(float(X_pca[mask, 1].mean()), 3)
            centroids_2d.append({"cluster": c_id, "x": c_x, "y": c_y})

    elapsed = round(time.time() - t0, 3)

    result = {
        "k": k,
        "n_samples": total_pts,
        "n_features": bundle["n_features"],
        "inertia": round(float(km.inertia_), 2),
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "elbow_curve": elbow_curve,
        "cluster_profiles": cluster_profiles,
        "scatter_points": scatter_pts,
        "centroids_2d": centroids_2d,
        "elapsed_sec": elapsed
    }
    _CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 2. HIERARCHICAL / AGGLOMERATIVE CLUSTERING
# ─────────────────────────────────────────────────────────

def run_hierarchical(n_clusters=3, linkage="ward"):
    """
    Executes Agglomerative Hierarchical Clustering with selectable linkage.
    Computes silhouette, cluster distribution, profiles, and 2D projection.
    """
    n_clusters = max(2, min(int(n_clusters), 6))
    if linkage not in ["ward", "complete", "average"]:
        linkage = "ward"

    cache_key = f"hierarchical_{n_clusters}_{linkage}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    t0 = time.time()
    bundle = _get_clean_data()
    # Hierarchical clustering has O(N^2) memory/time complexity, run on 1200 samples for subsecond execution
    limit = 1200
    X_sub = bundle["X_scaled"][:limit]
    raw_sub = bundle["raw_df"].iloc[:limit]
    pca_bundle = _get_pca_2d()
    X_pca = pca_bundle["X_pca"][:limit]

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X_sub)

    sil = round(float(silhouette_score(X_sub, labels)), 4)
    db = round(float(davies_bouldin_score(X_sub, labels)), 4)
    ch = round(float(calinski_harabasz_score(X_sub, labels)), 2)

    cluster_profiles = []
    total_pts = len(labels)
    for c_id in range(n_clusters):
        mask = (labels == c_id)
        sub_raw = raw_sub[mask]
        count = int(mask.sum())
        pct = round((count / total_pts) * 100, 1)

        mean_cgpa = round(float(sub_raw["CGPA"].mean()), 2) if "CGPA" in sub_raw else 0
        mean_att = round(float(sub_raw["AttendancePercent"].mean()), 1) if "AttendancePercent" in sub_raw else 0
        mean_code = round(float(sub_raw["CodingTestScore"].mean()), 1) if "CodingTestScore" in sub_raw else 0
        mean_apt = round(float(sub_raw["AptitudeTestScore"].mean()), 1) if "AptitudeTestScore" in sub_raw else 0
        place_rate = round(float((sub_raw["PlacementStatus"].astype(str).isin(["Placed", "1"])).mean() * 100), 1) if "PlacementStatus" in sub_raw else 0

        cluster_profiles.append({
            "cluster_id": c_id,
            "count": count,
            "pct": pct,
            "mean_cgpa": mean_cgpa,
            "mean_att": mean_att,
            "mean_code": mean_code,
            "mean_apt": mean_apt,
            "place_rate": place_rate
        })

    scatter_pts = []
    step = max(1, len(labels) // 600)
    for idx in range(0, len(labels), step):
        scatter_pts.append({
            "x": round(float(X_pca[idx, 0]), 3),
            "y": round(float(X_pca[idx, 1]), 3),
            "cluster": int(labels[idx]),
            "cgpa": round(float(raw_sub.iloc[idx]["CGPA"]), 2) if "CGPA" in raw_sub else 0,
            "placed": str(raw_sub.iloc[idx]["PlacementStatus"]) if "PlacementStatus" in raw_sub else "N/A"
        })

    elapsed = round(time.time() - t0, 3)

    result = {
        "n_clusters": n_clusters,
        "linkage": linkage,
        "n_samples": total_pts,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "cluster_profiles": cluster_profiles,
        "scatter_points": scatter_pts,
        "elapsed_sec": elapsed
    }
    _CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 3. DENSITY-BASED CLUSTERING (DBSCAN)
# ─────────────────────────────────────────────────────────

def run_dbscan(eps=0.7, min_samples=15):
    """
    Executes DBSCAN on 2D PCA manifold projection space to detect arbitrary dense
    clusters and isolate noise/anomaly candidates (label == -1).
    """
    eps = max(0.2, min(float(eps), 2.5))
    min_samples = max(5, min(int(min_samples), 50))

    cache_key = f"dbscan_eps{eps}_ms{min_samples}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    t0 = time.time()
    bundle = _get_clean_data()
    raw_df = bundle["raw_df"]
    pca_bundle = _get_pca_2d()
    X_pca = pca_bundle["X_pca"]

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_pca)

    unique_labels = sorted(list(set(labels)))
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = int((labels == -1).sum())
    total_pts = len(labels)
    noise_pct = round((n_noise / total_pts) * 100, 1)

    # Silhouette score on clustered points only (if >= 2 clusters)
    if n_clusters >= 2 and (labels != -1).sum() > n_clusters:
        clustered_mask = (labels != -1)
        sil = round(float(silhouette_score(X_pca[clustered_mask], labels[clustered_mask])), 4)
    else:
        sil = 0.0

    # Cluster summary
    cluster_profiles = []
    for c_id in unique_labels:
        mask = (labels == c_id)
        sub_raw = raw_df[mask]
        count = int(mask.sum())
        pct = round((count / total_pts) * 100, 1)

        is_noise = (c_id == -1)
        name = "Noise / Outliers (Low Density)" if is_noise else f"Dense Cluster {c_id}"

        mean_cgpa = round(float(sub_raw["CGPA"].mean()), 2) if "CGPA" in sub_raw else 0
        mean_att = round(float(sub_raw["AttendancePercent"].mean()), 1) if "AttendancePercent" in sub_raw else 0
        mean_code = round(float(sub_raw["CodingTestScore"].mean()), 1) if "CodingTestScore" in sub_raw else 0
        mean_apt = round(float(sub_raw["AptitudeTestScore"].mean()), 1) if "AptitudeTestScore" in sub_raw else 0
        place_rate = round(float((sub_raw["PlacementStatus"].astype(str).isin(["Placed", "1"])).mean() * 100), 1) if "PlacementStatus" in sub_raw else 0

        cluster_profiles.append({
            "cluster_id": c_id,
            "name": name,
            "is_noise": is_noise,
            "count": count,
            "pct": pct,
            "mean_cgpa": mean_cgpa,
            "mean_att": mean_att,
            "mean_code": mean_code,
            "mean_apt": mean_apt,
            "place_rate": place_rate
        })

    scatter_pts = []
    step = max(1, len(labels) // 750)
    for idx in range(0, len(labels), step):
        scatter_pts.append({
            "x": round(float(X_pca[idx, 0]), 3),
            "y": round(float(X_pca[idx, 1]), 3),
            "cluster": int(labels[idx]),
            "is_noise": bool(labels[idx] == -1),
            "cgpa": round(float(raw_df.iloc[idx]["CGPA"]), 2) if "CGPA" in raw_df else 0,
            "placed": str(raw_df.iloc[idx]["PlacementStatus"]) if "PlacementStatus" in raw_df else "N/A"
        })

    elapsed = round(time.time() - t0, 3)

    result = {
        "eps": eps,
        "min_samples": min_samples,
        "n_samples": total_pts,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "silhouette": sil,
        "cluster_profiles": cluster_profiles,
        "scatter_points": scatter_pts,
        "elapsed_sec": elapsed
    }
    _CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 4. DIMENSIONALITY REDUCTION: PCA
# ─────────────────────────────────────────────────────────

def run_pca():
    """
    Computes Principal Component Analysis (PCA):
    Explained variance ratios, cumulative variance, feature loadings,
    and 2D scatter coordinates.
    """
    cache_key = "pca_full_results"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    t0 = time.time()
    bundle = _get_clean_data()
    X = bundle["X_scaled"]
    raw_df = bundle["raw_df"]
    feature_names = bundle["feature_names"]

    pca_full = PCA(n_components=min(10, X.shape[1]), random_state=42)
    X_pca_full = pca_full.fit_transform(X)

    exp_var = [round(float(v) * 100, 2) for v in pca_full.explained_variance_ratio_]
    cum_var = [round(float(v) * 100, 2) for v in np.cumsum(pca_full.explained_variance_ratio_)]

    # Feature loadings for PC1 and PC2
    loadings_pc1 = pd.Series(pca_full.components_[0], index=feature_names)
    loadings_pc2 = pd.Series(pca_full.components_[1], index=feature_names)

    top_pc1_pos = [{"feature": k, "weight": round(float(v), 3)} for k, v in loadings_pc1.nlargest(5).items()]
    top_pc1_neg = [{"feature": k, "weight": round(float(v), 3)} for k, v in loadings_pc1.nsmallest(5).items()]
    top_pc2_pos = [{"feature": k, "weight": round(float(v), 3)} for k, v in loadings_pc2.nlargest(5).items()]
    top_pc2_neg = [{"feature": k, "weight": round(float(v), 3)} for k, v in loadings_pc2.nsmallest(5).items()]

    # 2D scatter points
    scatter_pts = []
    step = max(1, len(X) // 750)
    for idx in range(0, len(X), step):
        scatter_pts.append({
            "x": round(float(X_pca_full[idx, 0]), 3),
            "y": round(float(X_pca_full[idx, 1]), 3),
            "cgpa": round(float(raw_df.iloc[idx]["CGPA"]), 2) if "CGPA" in raw_df else 0,
            "placed": str(raw_df.iloc[idx]["PlacementStatus"]) if "PlacementStatus" in raw_df else "N/A"
        })

    elapsed = round(time.time() - t0, 3)

    result = {
        "n_samples": len(X),
        "n_features": len(feature_names),
        "pc1_var": exp_var[0],
        "pc2_var": exp_var[1],
        "pc3_var": exp_var[2] if len(exp_var) > 2 else 0,
        "total_2d_var": round(exp_var[0] + exp_var[1], 2),
        "explained_variance": exp_var,
        "cumulative_variance": cum_var,
        "top_pc1_pos": top_pc1_pos,
        "top_pc1_neg": top_pc1_neg,
        "top_pc2_pos": top_pc2_pos,
        "top_pc2_neg": top_pc2_neg,
        "scatter_points": scatter_pts,
        "elapsed_sec": elapsed
    }
    _CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 5. DIMENSIONALITY REDUCTION: t-SNE
# ─────────────────────────────────────────────────────────

def run_tsne(perplexity=30):
    """
    Computes t-Distributed Stochastic Neighbor Embedding (t-SNE) 2D projection.
    Reveals non-linear manifold groupings and localized student clusters.
    """
    perplexity = max(10, min(int(perplexity), 50))
    cache_key = f"tsne_perp{perplexity}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    t0 = time.time()
    bundle = _get_clean_data()
    # Run on 1,000 samples for subsecond interactive performance
    limit = 1000
    X_sub = bundle["X_scaled"][:limit]
    raw_sub = bundle["raw_df"].iloc[:limit]

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=600)
    X_tsne = tsne.fit_transform(X_sub)

    scatter_pts = []
    for idx in range(len(X_tsne)):
        scatter_pts.append({
            "x": round(float(X_tsne[idx, 0]), 3),
            "y": round(float(X_tsne[idx, 1]), 3),
            "cgpa": round(float(raw_sub.iloc[idx]["CGPA"]), 2) if "CGPA" in raw_sub else 0,
            "placed": str(raw_sub.iloc[idx]["PlacementStatus"]) if "PlacementStatus" in raw_sub else "N/A"
        })

    elapsed = round(time.time() - t0, 3)

    result = {
        "n_samples": limit,
        "perplexity": perplexity,
        "kl_divergence": round(float(getattr(tsne, "kl_divergence_", 0)), 3),
        "scatter_points": scatter_pts,
        "elapsed_sec": elapsed
    }
    _CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 6. DIMENSIONALITY REDUCTION: UMAP
# ─────────────────────────────────────────────────────────

def run_umap(n_neighbors=15, min_dist=0.1):
    """
    Uniform Manifold Approximation and Projection (UMAP).
    Preserves both fine local clustering and broad global topology with high speed.
    """
    n_neighbors = max(5, min(int(n_neighbors), 50))
    min_dist = max(0.01, min(float(min_dist), 0.99))

    cache_key = f"umap_nn{n_neighbors}_md{min_dist}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    t0 = time.time()
    bundle = _get_clean_data()
    limit = 800
    X_sub = bundle["X_scaled"][:limit]
    raw_sub = bundle["raw_df"].iloc[:limit]

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X_sub)

    scatter_pts = []
    for idx in range(len(X_umap)):
        scatter_pts.append({
            "x": round(float(X_umap[idx, 0]), 3),
            "y": round(float(X_umap[idx, 1]), 3),
            "cgpa": round(float(raw_sub.iloc[idx]["CGPA"]), 2) if "CGPA" in raw_sub else 0,
            "placed": str(raw_sub.iloc[idx]["PlacementStatus"]) if "PlacementStatus" in raw_sub else "N/A"
        })

    elapsed = round(time.time() - t0, 3)

    result = {
        "n_samples": limit,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "scatter_points": scatter_pts,
        "elapsed_sec": elapsed
    }
    _CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 7. ANOMALY DETECTION: ISOLATION FOREST
# ─────────────────────────────────────────────────────────

def run_anomaly_detection(contamination=0.05):
    """
    Detects outliers and abnormal student patterns using Isolation Forest.
    Returns:
    - Inlier / Outlier counts & percentages
    - Normal vs Anomaly profile comparison
    - Top anomalous student records with explanations
    - 2D PCA scatter points with anomaly highlighting
    """
    contamination = max(0.01, min(float(contamination), 0.20))
    cache_key = f"anomaly_contam{contamination}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    t0 = time.time()
    bundle = _get_clean_data()
    X = bundle["X_scaled"]
    raw_df = bundle["raw_df"].copy()
    pca_bundle = _get_pca_2d()
    X_pca = pca_bundle["X_pca"]

    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X)
    scores = iso.decision_function(X)  # lower = more abnormal

    raw_df["_anomaly"] = (preds == -1)
    raw_df["_anomaly_score"] = scores

    n_anomalies = int((preds == -1).sum())
    n_normal = int((preds == 1).sum())
    total_pts = len(preds)

    # Comparative Profiles: Normal vs Anomaly
    normal_df = raw_df[raw_df["_anomaly"] == False]
    anomaly_df = raw_df[raw_df["_anomaly"] == True]

    def _calc_profile(df_subset):
        mean_cgpa = round(float(df_subset["CGPA"].mean()), 2) if "CGPA" in df_subset else 0
        mean_att = round(float(df_subset["AttendancePercent"].mean()), 1) if "AttendancePercent" in df_subset else 0
        mean_code = round(float(df_subset["CodingTestScore"].mean()), 1) if "CodingTestScore" in df_subset else 0
        mean_apt = round(float(df_subset["AptitudeTestScore"].mean()), 1) if "AptitudeTestScore" in df_subset else 0
        mean_soft = round(float(df_subset["SoftSkillsRating"].mean()), 2) if "SoftSkillsRating" in df_subset else 0
        mean_proj = round(float(df_subset["Projects"].mean()), 1) if "Projects" in df_subset else 0
        mean_intern = round(float(df_subset["Internships"].mean()), 1) if "Internships" in df_subset else 0
        
        if "HistoryOfBacklogs" in df_subset:
            backlog_rate = round(float((df_subset["HistoryOfBacklogs"].astype(str).str.lower().isin(["yes", "1"])).mean() * 100), 1)
        else:
            backlog_rate = 0.0

        if "PlacementStatus" in df_subset:
            place_rate = round(float((df_subset["PlacementStatus"].astype(str).isin(["Placed", "1"])).mean() * 100), 1)
        else:
            place_rate = 0.0

        return {
            "mean_cgpa": mean_cgpa,
            "mean_att": mean_att,
            "mean_code": mean_code,
            "mean_apt": mean_apt,
            "mean_soft": mean_soft,
            "mean_proj": mean_proj,
            "mean_intern": mean_intern,
            "backlog_rate": backlog_rate,
            "place_rate": place_rate
        }

    normal_profile = _calc_profile(normal_df)
    anomaly_profile = _calc_profile(anomaly_df)

    # Extract Top 10 Anomalous Students sorted by lowest score
    top_anomalies_raw = anomaly_df.sort_values(by="_anomaly_score", ascending=True).head(10)
    top_anomalies = []
    for _, row in top_anomalies_raw.iterrows():
        # Formulate human-readable explanation of why this record is an anomaly
        reasons = []
        cgpa_val = float(row.get("CGPA", 0))
        att_val = float(row.get("AttendancePercent", 0))
        code_val = float(row.get("CodingTestScore", 0))
        backlogs_val = str(row.get("HistoryOfBacklogs", "No")).lower() in ["yes", "1"]
        intern_val = int(row.get("Internships", 0))

        if cgpa_val > 9.0 and att_val < 65:
            reasons.append("Very high CGPA despite critical low attendance")
        elif cgpa_val < 6.0 and code_val > 85:
            reasons.append("Low CGPA paired with top-tier coding score")
        elif backlogs_val and intern_val >= 3:
            reasons.append("Backlogs present but unusually high internships")
        elif code_val > 90 and float(row.get("AptitudeTestScore", 0)) < 40:
            reasons.append("Severe divergence: High coding vs weak aptitude")
        else:
            reasons.append("Atypical combination of academic and test metrics")

        top_anomalies.append({
            "student_id": str(row.get("StudentID", "N/A")),
            "score": round(float(row["_anomaly_score"]), 4),
            "cgpa": cgpa_val,
            "attendance": att_val,
            "coding": code_val,
            "aptitude": float(row.get("AptitudeTestScore", 0)),
            "projects": int(row.get("Projects", 0)),
            "internships": intern_val,
            "backlogs": "Yes" if backlogs_val else "No",
            "placement": str(row.get("PlacementStatus", "N/A")),
            "reason": reasons[0] if reasons else "Extreme outlier in multi-dimensional feature space"
        })

    # Scatter points for 2D visualization
    scatter_pts = []
    step = max(1, len(preds) // 750)
    for idx in range(0, len(preds), step):
        scatter_pts.append({
            "x": round(float(X_pca[idx, 0]), 3),
            "y": round(float(X_pca[idx, 1]), 3),
            "is_anomaly": bool(preds[idx] == -1),
            "score": round(float(scores[idx]), 3),
            "cgpa": round(float(raw_df.iloc[idx]["CGPA"]), 2) if "CGPA" in raw_df else 0,
            "placed": str(raw_df.iloc[idx]["PlacementStatus"]) if "PlacementStatus" in raw_df else "N/A"
        })

    elapsed = round(time.time() - t0, 3)

    result = {
        "contamination": contamination,
        "n_samples": total_pts,
        "n_anomalies": n_anomalies,
        "n_normal": n_normal,
        "anomaly_pct": round((n_anomalies / total_pts) * 100, 1),
        "normal_profile": normal_profile,
        "anomaly_profile": anomaly_profile,
        "top_anomalies": top_anomalies,
        "scatter_points": scatter_pts,
        "elapsed_sec": elapsed
    }
    _CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 8. BENCHMARK & COMPARISON OVERVIEW
# ─────────────────────────────────────────────────────────

def run_comparison():
    """
    Returns comparative synthesis matrices for all clustering algorithms
    and all dimensionality reduction techniques.
    """
    km_res = run_kmeans(k=3)
    hier_res = run_hierarchical(n_clusters=3, linkage="ward")
    db_res = run_dbscan(eps=0.7, min_samples=15)
    pca_res = run_pca()
    anom_res = run_anomaly_detection(contamination=0.05)

    clustering_table = [
        {
            "method": "K-Means",
            "clusters": km_res["k"],
            "silhouette": km_res["silhouette"],
            "davies_bouldin": km_res["davies_bouldin"],
            "complexity": "O(k · n · d) — Very Fast",
            "noise_handling": "Forces every point into a cluster (Sensitive to outliers)",
            "geometry": "Convex / Spherical clusters",
            "best_for": "Large datasets, clear spherical clusters, fast segmentation"
        },
        {
            "method": "Hierarchical (Ward)",
            "clusters": hier_res["n_clusters"],
            "silhouette": hier_res["silhouette"],
            "davies_bouldin": hier_res["davies_bouldin"],
            "complexity": "O(n² · d) — Computationally Heavy",
            "noise_handling": "No native noise rejection",
            "geometry": "Tree-structured dendrogram hierarchies",
            "best_for": "Small to medium datasets, nested taxonomies, explainable hierarchy"
        },
        {
            "method": "DBSCAN",
            "clusters": db_res["n_clusters"],
            "silhouette": db_res["silhouette"],
            "davies_bouldin": "N/A (Density based)",
            "complexity": "O(n log n) with spatial index",
            "noise_handling": f"Isolates true noise ({db_res['noise_pct']}% identified)",
            "geometry": "Arbitrary shapes, non-convex density basins",
            "best_for": "Irregular spatial distributions and outlier-heavy data"
        }
    ]

    reduction_table = [
        {
            "method": "PCA",
            "type": "Linear Projection",
            "speed": f"{pca_res['elapsed_sec']}s — Instantaneous",
            "structure_preserved": "Global variance (preserves maximum variance across all axes)",
            "interpretability": "High — Direct feature loadings and principal vectors",
            "use_case": "Feature compression, multi-collinearity removal, baseline 2D plotting"
        },
        {
            "method": "t-SNE",
            "type": "Non-linear Manifold",
            "speed": "Moderate (~1-2s)",
            "structure_preserved": "Local neighborhood affinities and distinct sub-populations",
            "interpretability": "Low — Distances between distant clusters are not preserved",
            "use_case": "Visualizing complex high-dimensional cluster boundaries"
        },
        {
            "method": "UMAP",
            "type": "Non-linear Manifold (Fuzzy Simplicial)",
            "speed": "Fast (~2-3s)",
            "structure_preserved": "Balances local cluster cohesion with global macro-topology",
            "interpretability": "Medium — Preserves relative continuum between clusters",
            "use_case": "Modern standard for high-dimensional single-cell & ML manifold visualization"
        }
    ]

    return {
        "clustering_table": clustering_table,
        "reduction_table": reduction_table,
        "kmeans_preview": km_res,
        "anomaly_preview": anom_res
    }
