"""
feature_scaling.py
==================
Feature Scaling & Standardisation module.
Covers Min-Max, Standard (Z-score), and Robust scaling
applied to key numeric columns of the placement dataset.
"""

import pandas as pd
from load_data import load_data

# Numeric columns to scale (as used in the notebooks)
SCALE_COLUMNS = [
    "CGPA",
    "AttendancePercent",
    "AptitudeTestScore",
    "CodingTestScore",
    "SoftSkillsRating",
    "MockInterviewScore",
    "Salary Package",
]


def _available_cols(df):
    """Return only those SCALE_COLUMNS that actually exist in df."""
    return [c for c in SCALE_COLUMNS if c in df.columns]


def _preview(df, cols, n=10):
    """Return first n rows of selected columns as list-of-dicts."""
    return df[cols].head(n).round(4).to_dict(orient="records")


# ─────────────────────────────────────────
# Original Stats
# ─────────────────────────────────────────

def get_original_stats():
    """
    Returns raw (before scaling) stats for each scale column:
    min, max, mean, std, median.
    """
    df = load_data()
    cols = _available_cols(df)
    stats = []
    for col in cols:
        s = df[col].dropna()
        stats.append({
            "column": col,
            "min":    round(float(s.min()),  4),
            "max":    round(float(s.max()),  4),
            "mean":   round(float(s.mean()), 4),
            "std":    round(float(s.std()),  4),
            "median": round(float(s.median()), 4),
        })
    return {"columns": cols, "stats": stats}


# ─────────────────────────────────────────
# Min-Max Scaling  (Normalisation)
# ─────────────────────────────────────────

def minmax_scaling():
    """
    Applies Min-Max scaling:  x_new = (x - min) / (max - min)
    Output range: [0, 1]
    Returns: stats before/after, preview rows, formula, column list.
    """
    df = load_data()
    cols = _available_cols(df)

    scaled_df = df.copy()
    stats_before = []
    stats_after  = []
    col_mins     = {}
    col_maxs     = {}

    for col in cols:
        s = scaled_df[col].dropna()
        mn, mx = float(s.min()), float(s.max())
        col_mins[col] = round(mn, 4)
        col_maxs[col] = round(mx, 4)
        stats_before.append({
            "column": col,
            "min": round(mn, 4), "max": round(mx, 4),
            "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
        })
        denom = mx - mn if (mx - mn) != 0 else 1
        scaled_df[col] = (scaled_df[col] - mn) / denom
        sa = scaled_df[col].dropna()
        stats_after.append({
            "column": col,
            "min": round(float(sa.min()), 4), "max": round(float(sa.max()), 4),
            "mean": round(float(sa.mean()), 4), "std": round(float(sa.std()), 4),
        })

    return {
        "method":        "Min-Max Scaling (Normalization)",
        "formula":       "x_new = (x - min) / (max - min)",
        "output_range":  "[0, 1]",
        "use_case":      "When you need values in a fixed range [0,1]. Used in neural networks, KNN, SVM.",
        "columns":       cols,
        "stats_before":  stats_before,
        "stats_after":   stats_after,
        "preview_before": _preview(df, cols),
        "preview_after":  _preview(scaled_df, cols),
    }


# ─────────────────────────────────────────
# Standard Scaling  (Z-score / Standardisation)
# ─────────────────────────────────────────

def standard_scaling():
    """
    Applies Standard scaling:  x_new = (x - mean) / std
    Output: mean ≈ 0, std ≈ 1
    """
    df = load_data()
    cols = _available_cols(df)

    scaled_df = df.copy()
    stats_before = []
    stats_after  = []

    for col in cols:
        s = scaled_df[col].dropna()
        mn, sd = float(s.mean()), float(s.std())
        stats_before.append({
            "column": col,
            "min":  round(float(s.min()), 4), "max": round(float(s.max()), 4),
            "mean": round(mn, 4),             "std": round(sd, 4),
        })
        if sd == 0:
            sd = 1
        scaled_df[col] = (scaled_df[col] - mn) / sd
        sa = scaled_df[col].dropna()
        stats_after.append({
            "column": col,
            "min":  round(float(sa.min()),  4), "max": round(float(sa.max()),  4),
            "mean": round(float(sa.mean()), 4), "std": round(float(sa.std()),  4),
        })

    return {
        "method":        "Standard Scaling (Z-score / Standardisation)",
        "formula":       "x_new = (x - mean) / std",
        "output_range":  "mean = 0, std = 1 (unbounded)",
        "use_case":      "Most common choice. Used in linear models, logistic regression, PCA, SVM.",
        "columns":       cols,
        "stats_before":  stats_before,
        "stats_after":   stats_after,
        "preview_before": _preview(df, cols),
        "preview_after":  _preview(scaled_df, cols),
    }


# ─────────────────────────────────────────
# Robust Scaling
# ─────────────────────────────────────────

def robust_scaling():
    """
    Applies Robust scaling:  x_new = (x - median) / IQR
    Resistant to outliers (uses median and IQR instead of mean/std).
    """
    df = load_data()
    cols = _available_cols(df)

    scaled_df = df.copy()
    stats_before = []
    stats_after  = []

    for col in cols:
        s   = scaled_df[col].dropna()
        med = float(s.median())
        q1  = float(s.quantile(0.25))
        q3  = float(s.quantile(0.75))
        iqr = q3 - q1 if (q3 - q1) != 0 else 1
        stats_before.append({
            "column": col,
            "median": round(med, 4), "iqr":  round(iqr, 4),
            "q1":     round(q1,  4), "q3":   round(q3, 4),
            "min":    round(float(s.min()), 4), "max": round(float(s.max()), 4),
        })
        scaled_df[col] = (scaled_df[col] - med) / iqr
        sa = scaled_df[col].dropna()
        stats_after.append({
            "column": col,
            "median": round(float(sa.median()), 4), "iqr":  round(float(sa.quantile(0.75) - sa.quantile(0.25)), 4),
            "q1":     round(float(sa.quantile(0.25)), 4),     "q3":   round(float(sa.quantile(0.75)), 4),
            "min":    round(float(sa.min()), 4), "max": round(float(sa.max()), 4),
        })

    return {
        "method":        "Robust Scaling",
        "formula":       "x_new = (x - median) / IQR",
        "output_range":  "Centered at 0, scale = 1 IQR unit (unbounded, outlier-resistant)",
        "use_case":      "Use when data has significant outliers. Robust to extreme values.",
        "columns":       cols,
        "stats_before":  stats_before,
        "stats_after":   stats_after,
        "preview_before": _preview(df, cols),
        "preview_after":  _preview(scaled_df, cols),
    }


# ─────────────────────────────────────────
# Comparison summary  (all three methods)
# ─────────────────────────────────────────

def get_scaling_comparison():
    """
    Returns a comparison dict showing CGPA before and after each method.
    """
    df = load_data()
    col = "CGPA" if "CGPA" in df.columns else _available_cols(df)[0]
    s   = df[col].dropna()

    mn, mx  = float(s.min()), float(s.max())
    mean_v  = float(s.mean())
    std_v   = float(s.std()) or 1.0
    med_v   = float(s.median())
    q1, q3  = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr_v   = (q3 - q1) or 1.0

    mm     = (s - mn) / (mx - mn)
    zscore = (s - mean_v) / std_v
    robust = (s - med_v) / iqr_v

    rows = []
    for i in range(min(10, len(s))):
        rows.append({
            "Original":     round(float(s.iloc[i]), 4),
            "Min-Max":      round(float(mm.iloc[i]), 4),
            "Z-score":      round(float(zscore.iloc[i]), 4),
            "Robust":       round(float(robust.iloc[i]), 4),
        })
    return {"column": col, "rows": rows}
