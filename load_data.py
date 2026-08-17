import os
import numpy as np
import pandas as pd

DATA_PATH = r"D:\2-1 Odd sem\Machine Learning\Placementpredictor\placement_predict_50k Dataset.csv"


# ─────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


# ─────────────────────────────────────────
# Data Loading page functions
# ─────────────────────────────────────────

def get_data_summary(path: str = DATA_PATH) -> dict:
    df = load_data(path)
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        "preview": df.head(10).to_dict(orient="records"),
    }
    return summary


def get_duplicate_count(path: str = DATA_PATH) -> int:
    df = load_data(path)
    return int(df.duplicated().sum())


# ─────────────────────────────────────────
# EDA summary — overview stats
# ─────────────────────────────────────────

def get_eda_summary(path: str = DATA_PATH) -> dict:
    df = load_data(path)
    total = len(df)

    # Target distribution
    placed_count = 0
    not_placed_count = 0
    placed_pct = 0.0

    if "PlacementStatus" in df.columns:
        vc = df["PlacementStatus"].value_counts().to_dict()
        placed_count    = int(vc.get(1, vc.get("Placed", vc.get("1", 0))))
        not_placed_count = int(vc.get(0, vc.get("Not Placed", vc.get("0", total - placed_count))))
        placed_pct = round((placed_count / total) * 100, 1) if total else 0.0

    # Stream distribution
    stream_dist = {}
    if "Stream" in df.columns:
        stream_dist = {str(k): int(v) for k, v in df["Stream"].value_counts().head(6).items()}

    # Descriptive stats for key numeric columns
    priority_cols = [
        "CGPA", "AttendancePercent", "AptitudeTestScore",
        "CodingTestScore", "SoftSkillsRating", "MockInterviewScore",
        "Internships", "Projects", "Workshops"
    ]
    avail_cols = [c for c in priority_cols if c in df.columns]
    stats_table = []
    if avail_cols:
        desc = df[avail_cols].describe().round(2)
        for col in avail_cols:
            stats_table.append({
                "feature": col,
                "mean":   float(desc.loc["mean", col]),
                "std":    float(desc.loc["std",  col]),
                "min":    float(desc.loc["min",  col]),
                "median": float(desc.loc["50%",  col]),
                "max":    float(desc.loc["max",  col]),
            })

    return {
        "total_records":   total,
        "placed_count":    placed_count,
        "not_placed_count": not_placed_count,
        "placed_pct":      placed_pct,
        "not_placed_pct":  round(100 - placed_pct, 1),
        "stream_dist":     stream_dist,
        "stats_table":     stats_table,
        "avg_cgpa":        round(float(df["CGPA"].mean()), 2)           if "CGPA"            in df.columns else 0,
        "avg_coding":      round(float(df["CodingTestScore"].mean()), 1) if "CodingTestScore"  in df.columns else 0,
        "avg_aptitude":    round(float(df["AptitudeTestScore"].mean()), 1) if "AptitudeTestScore" in df.columns else 0,
    }


# ─────────────────────────────────────────
# Column metadata  (for dropdowns)
# ─────────────────────────────────────────

def get_columns_metadata(path: str = DATA_PATH) -> dict:
    df = load_data(path)
    return {
        "all":         list(df.columns),
        "numerical":   list(df.select_dtypes(include=["number"]).columns),
        "categorical": list(df.select_dtypes(exclude=["number"]).columns),
    }


# ─────────────────────────────────────────
# Plot data  — returns JSON-serialisable dict
# consumed by /api/plot-data → Plotly (JS)
# ─────────────────────────────────────────

def _safe_sample(series, n, seed=42):
    """Sample up to n rows from a Series safely."""
    n = min(n, len(series))
    return series.sample(n, random_state=seed) if n > 0 else series


def get_plot_data(column: str = "CGPA", chart_type: str = "histogram",
                  path: str = DATA_PATH) -> dict:
    df = load_data(path)

    # Default column fallback
    if column not in df.columns and chart_type != "heatmap":
        num_cols = list(df.select_dtypes(include="number").columns)
        column = num_cols[0] if num_cols else df.columns[0]

    is_num = column in df.select_dtypes(include="number").columns
    has_target = "PlacementStatus" in df.columns

    result = {
        "column":        column,
        "chart_type":    chart_type,
        "is_numerical":  bool(is_num),
        "data":          {},
        "stats":         {},
        "insight":       ""
    }

    # ── 1. CORRELATION HEATMAP ───────────────────────────────────
    if chart_type == "heatmap":
        num_df = df.select_dtypes(include="number")
        priority = [
            "CGPA", "SGPA_Sem7", "SGPA_Sem8", "AttendancePercent",
            "Internships", "Projects", "Certifications",
            "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating",
            "MockInterviewScore", "PlacementStatus", "Salary Package"
        ]
        active = [c for c in priority if c in num_df.columns] or list(num_df.columns)
        corr   = num_df[active].corr().round(2)
        result["data"]  = {"columns": active, "z": corr.values.tolist()}
        result["stats"] = {
            "Features in Matrix": len(active),
            "Total Numeric Cols": len(num_df.columns),
        }
        result["insight"] = "Correlation matrix showing linear relationships between all numeric features."
        return result

    # ── 2. HISTOGRAM / DISTRIBUTION ─────────────────────────────
    if chart_type == "histogram":
        if not is_num:
            return get_plot_data(column, "bar", path)
        series = df[column].dropna()
        mean_v   = float(series.mean())
        median_v = float(series.median())
        std_v    = float(series.std())
        skew_v   = float(series.skew())
        sample   = _safe_sample(series, 2000).tolist()

        placed_vals     = []
        not_placed_vals = []
        if has_target:
            p_series  = df[df["PlacementStatus"] == 1][column].dropna()
            np_series = df[df["PlacementStatus"] == 0][column].dropna()
            placed_vals     = _safe_sample(p_series,  1000).tolist()
            not_placed_vals = _safe_sample(np_series, 1000).tolist()

        result["data"]  = {"sample_values": sample, "placed_sample": placed_vals, "not_placed_sample": not_placed_vals}
        result["stats"] = {
            "Mean":      round(mean_v, 2),
            "Median":    round(median_v, 2),
            "Std Dev":   round(std_v, 2),
            "Skewness":  round(skew_v, 2),
            "Min – Max": f"{round(float(series.min()), 2)} – {round(float(series.max()), 2)}",
            "Count":     len(series),
        }
        result["insight"] = (
            f"{column} has mean {round(mean_v, 2)}, median {round(median_v, 2)}, "
            f"and skewness of {round(skew_v, 2)}."
        )

    # ── 3. BAR / COUNT PLOT ─────────────────────────────────────
    elif chart_type == "bar":
        counts     = df[column].value_counts().head(15)
        categories = [str(k) for k in counts.index]
        values     = counts.values.tolist()
        total      = int(df[column].dropna().__len__())
        result["data"]  = {
            "categories":  categories,
            "values":      values,
            "percentages": [round((v / total) * 100, 1) for v in values],
        }
        result["stats"] = {
            "Top Category":   f"{categories[0]} ({values[0]:,})" if categories else "N/A",
            "Unique Classes": int(df[column].nunique()),
            "Total Entries":  total,
        }
        result["insight"] = (
            f"Dominant category in {column} is '{categories[0]}' "
            f"({round((values[0]/total)*100, 1) if categories else 0}%)."
        ) if categories else ""

    # ── 4. BOX PLOT ─────────────────────────────────────────────
    elif chart_type == "box":
        if not is_num:
            return get_plot_data(column, "bar", path)
        series = df[column].dropna()
        q1  = float(series.quantile(0.25))
        med = float(series.median())
        q3  = float(series.quantile(0.75))
        iqr = q3 - q1
        outliers = int(((series < (q1 - 1.5*iqr)) | (series > (q3 + 1.5*iqr))).sum())

        placed_s, not_placed_s = [], []
        if has_target:
            placed_s     = _safe_sample(df[df["PlacementStatus"] == 1][column].dropna(), 1200).tolist()
            not_placed_s = _safe_sample(df[df["PlacementStatus"] == 0][column].dropna(), 1200).tolist()

        result["data"]  = {
            "placed_sample":     placed_s,
            "not_placed_sample": not_placed_s,
            "all_sample":        _safe_sample(series, 1500).tolist(),
        }
        result["stats"] = {
            "Q1 (25th %)":    round(q1, 2),
            "Median (50th %)": round(med, 2),
            "Q3 (75th %)":    round(q3, 2),
            "IQR":            round(iqr, 2),
            "Outliers":       outliers,
        }
        result["insight"] = f"Middle 50% of {column} spans {round(q1,2)} – {round(q3,2)} (IQR = {round(iqr,2)})."

    # ── 5. SCATTER ──────────────────────────────────────────────
    elif chart_type == "scatter":
        if not is_num:
            return get_plot_data(column, "bar", path)
        num_cols = list(df.select_dtypes(include="number").columns)
        target_y = next((c for c in num_cols if c != column), column)
        cols_needed = [c for c in [column, target_y, "PlacementStatus"] if c in df.columns]
        sample_df = df[cols_needed].dropna()
        sample_df = _safe_sample(sample_df, 1000)
        placement_col = sample_df["PlacementStatus"].tolist() if "PlacementStatus" in sample_df else []
        corr_val = round(float(df[column].corr(df[target_y])), 2) if target_y in df.columns else 0.0

        result["data"]  = {
            "x":         sample_df[column].tolist(),
            "y":         sample_df[target_y].tolist(),
            "y_col":     target_y,
            "placement": placement_col,
        }
        result["stats"] = {
            "X-Axis":          column,
            "Y-Axis":          target_y,
            "Correlation (r)": corr_val,
            "Sample Points":   len(sample_df),
        }
        result["insight"] = f"Correlation between {column} and {target_y}: r = {corr_val}."

    # ── 6. VIOLIN ───────────────────────────────────────────────
    elif chart_type == "violin":
        if not is_num:
            return get_plot_data(column, "bar", path)
        placed_s, not_placed_s = [], []
        if has_target:
            placed_s     = _safe_sample(df[df["PlacementStatus"] == 1][column].dropna(), 1200).tolist()
            not_placed_s = _safe_sample(df[df["PlacementStatus"] == 0][column].dropna(), 1200).tolist()
        p_mean  = round(float(df[df["PlacementStatus"] == 1][column].mean()), 2) if has_target else "N/A"
        np_mean = round(float(df[df["PlacementStatus"] == 0][column].mean()), 2) if has_target else "N/A"

        result["data"]  = {"placed_sample": placed_s, "not_placed_sample": not_placed_s}
        result["stats"] = {
            "Placed Mean":     p_mean,
            "Not Placed Mean": np_mean,
            "Feature":         column,
        }
        result["insight"] = f"Violin plot shows density differences in {column} across placement outcomes."

    # ── 7. CDF (LINE) ───────────────────────────────────────────
    elif chart_type == "line":
        if not is_num:
            return get_plot_data(column, "bar", path)
        series = df[column].dropna().sort_values()
        n      = len(series)
        step   = max(1, n // 300)
        sub    = series.iloc[::step]
        cdf_y  = [round((i / n) * 100, 2) for i in range(0, n, step)]

        result["data"]  = {"x": sub.tolist(), "y": cdf_y}
        result["stats"] = {
            "50th Percentile": round(float(series.quantile(0.50)), 2),
            "90th Percentile": round(float(series.quantile(0.90)), 2),
            "Total Samples":   n,
        }
        result["insight"] = f"CDF shows percentile thresholds for {column}."

    return result


# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Summary:     ", get_data_summary())
    print("Duplicates:  ", get_duplicate_count())
    print("Columns Meta:", get_columns_metadata())
