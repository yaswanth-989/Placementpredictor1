"""
encoding.py
===========
Categorical Feature Encoding module.
Covers Label Encoding, One-Hot Encoding, and Ordinal Encoding
applied to categorical columns of the placement dataset.
"""

import pandas as pd
from load_data import load_data

# Categorical columns to encode
CAT_COLUMNS = [
    "Gender",
    "Stream",
    "City",
    "CollegeTier",
    "CGPA_Tier",
    "PlacementStatus",
]

# Ordinal column order mappings (for columns with a meaningful order)
ORDINAL_MAPS = {
    "CGPA_Tier":   ["Low", "Mid", "High"],
    "CollegeTier": ["Tier3", "Tier2", "Tier1"],
}


def _available_cat_cols(df):
    """Return only those CAT_COLUMNS that actually exist in df."""
    return [c for c in CAT_COLUMNS if c in df.columns]


def _preview(df, cols, n=10):
    return df[cols].head(n).to_dict(orient="records")


# ─────────────────────────────────────────
# Column Info
# ─────────────────────────────────────────

def get_categorical_info():
    """Returns info about each categorical column (unique values, top value, dtype)."""
    df = load_data()
    cols = _available_cat_cols(df)
    info = []
    for col in cols:
        vc = df[col].value_counts()
        info.append({
            "column":      col,
            "dtype":       str(df[col].dtype),
            "unique":      int(df[col].nunique()),
            "top_value":   str(vc.index[0]) if len(vc) > 0 else "-",
            "top_freq":    int(vc.iloc[0])   if len(vc) > 0 else 0,
            "categories":  [str(v) for v in vc.index.tolist()[:10]],
        })
    return {"columns": cols, "info": info}


# ─────────────────────────────────────────
# Label Encoding
# ─────────────────────────────────────────

def label_encoding():
    """
    Label Encoding: assigns a unique integer to each category alphabetically.
    e.g. Female→0, Male→1 | CSE→0, ECE→1, IT→2 …
    Warning: imposes arbitrary ordinal relationship on nominal categories.
    """
    df = load_data()
    cols = _available_cat_cols(df)

    encoded_df = df.copy()
    mappings   = {}

    for col in cols:
        unique_vals  = sorted(df[col].dropna().astype(str).unique())
        mapping      = {v: i for i, v in enumerate(unique_vals)}
        mappings[col] = mapping
        encoded_df[col] = df[col].astype(str).map(mapping)

    # Build display table: column, category → code
    mapping_table = []
    for col, m in mappings.items():
        for cat, code in m.items():
            mapping_table.append({"column": col, "category": cat, "code": code})

    return {
        "method":        "Label Encoding",
        "description":   "Maps each unique category to an integer (0, 1, 2 ...). "
                         "Simple but imposes an implicit ordinal relationship — "
                         "use only for ordinal features or tree-based models.",
        "warning":       "Do NOT use for nominal features (Gender, City, Stream) "
                         "with linear models — the arbitrary numbers imply a meaningless order.",
        "columns":       cols,
        "mappings":      mappings,
        "mapping_table": mapping_table,
        "preview_before": _preview(df, cols),
        "preview_after":  _preview(encoded_df, cols),
    }


# ─────────────────────────────────────────
# One-Hot Encoding
# ─────────────────────────────────────────

def onehot_encoding(max_categories=6):
    """
    One-Hot Encoding: creates a binary column for each unique category.
    Avoids the ordinal problem of label encoding.
    Drops the first column to avoid the dummy-variable trap.
    Only encodes columns with ≤ max_categories unique values to avoid explosion.
    """
    df = load_data()
    cols = _available_cat_cols(df)

    # Only encode low-cardinality columns
    eligible = [c for c in cols if df[c].nunique() <= max_categories]

    encoded_df = df.copy()
    ohe_info   = []

    for col in eligible:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        encoded_df = pd.concat([encoded_df.drop(columns=[col]), dummies], axis=1)
        ohe_info.append({
            "column":      col,
            "categories":  df[col].dropna().astype(str).unique().tolist(),
            "new_columns": dummies.columns.tolist(),
        })

    # Preview: show original cols + new encoded cols
    new_cols = [c for info in ohe_info for c in info["new_columns"]]
    preview_cols = new_cols[:12]   # limit display width

    return {
        "method":      "One-Hot Encoding",
        "description": "Creates a binary (0/1) column for each category. "
                       "Safe for nominal features. Drops first column to avoid the dummy trap.",
        "note":        f"Only columns with ≤ {max_categories} unique values are encoded "
                       f"(high-cardinality columns like City/Stream may be skipped or label-encoded instead).",
        "eligible":    eligible,
        "skipped":     [c for c in cols if c not in eligible],
        "ohe_info":    ohe_info,
        "new_columns": new_cols,
        "preview_before": _preview(df, eligible if eligible else cols),
        "preview_after":  _preview(encoded_df, preview_cols) if preview_cols else [],
    }


# ─────────────────────────────────────────
# Ordinal Encoding
# ─────────────────────────────────────────

def ordinal_encoding():
    """
    Ordinal Encoding: manually maps ordered categories to integers
    respecting their actual rank (Low<Mid<High, Tier3<Tier2<Tier1).
    Only applied to columns with a known ordering.
    """
    df = load_data()

    # Only encode columns that have a defined order AND exist in df
    cols = [c for c in ORDINAL_MAPS if c in df.columns]

    encoded_df = df.copy()
    encoding_info = []

    for col in cols:
        order = ORDINAL_MAPS[col]
        mapping = {v: i for i, v in enumerate(order)}
        encoded_df[col] = df[col].astype(str).map(mapping)
        # Check placement stats per category
        placement_stats = []
        if "PlacementStatus" in df.columns:
            for cat in order:
                subset = df[df[col] == cat]
                rate   = round(float(subset["PlacementStatus"].mean()) * 100, 1) \
                         if len(subset) > 0 and pd.api.types.is_numeric_dtype(df["PlacementStatus"]) else "-"
                placement_stats.append({"category": cat, "count": len(subset), "placement_rate_pct": rate})

        encoding_info.append({
            "column":          col,
            "order":           order,
            "mapping":         mapping,
            "placement_stats": placement_stats,
        })

    return {
        "method":      "Ordinal Encoding",
        "description": "Manually assigns ordered integers that reflect the real rank of categories. "
                       "Correct for features with a meaningful hierarchy (Low < Mid < High).",
        "note":        "Always verify the order against the target (placement rate). "
                       "The table below confirms the order is meaningful.",
        "columns":     cols,
        "encoding_info": encoding_info,
        "preview_before": _preview(df, cols) if cols else [],
        "preview_after":  _preview(encoded_df, cols) if cols else [],
    }


# ─────────────────────────────────────────
# Summary: frequency counts per column
# ─────────────────────────────────────────

def get_encoding_summary():
    """Returns frequency distribution for each categorical column."""
    df = load_data()
    cols = _available_cat_cols(df)
    summary = {}
    for col in cols:
        vc = df[col].value_counts()
        summary[col] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
    return summary
