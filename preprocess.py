import pandas as pd
from load_data import load_data

# Columns that have missing values and are targeted for imputation
NUMERIC_COLUMNS = [
    'Workshops',
    'AptitudeTestScore',
    'SoftSkillsRating',
    'CodingTestScore',
    'MockInterviewScore'
]


# ─────────────────────────────────────────
# Missing Value Summary
# ─────────────────────────────────────────

def get_missing_summary():
    """
    Returns a list of dicts: one per column,
    with missing count and missing percentage.
    """
    df = load_data()
    total = len(df)
    result = []
    for col in df.columns:
        count = int(df[col].isnull().sum())
        pct = round((count / total) * 100, 2)
        result.append({
            "column": col,
            "missing_count": count,
            "missing_pct": pct
        })
    return result


# ─────────────────────────────────────────
# Drop Missing Values
# ─────────────────────────────────────────

def drop_missing():
    """
    Drops all rows with any missing values.
    Returns before/after row counts and number of rows dropped.
    """
    df = load_data()
    before = len(df)
    df_dropped = df.dropna()
    after = len(df_dropped)
    return {
        "before": before,
        "after": after,
        "dropped": before - after
    }


# ─────────────────────────────────────────
# Mean Imputation
# ─────────────────────────────────────────

def mean_imputation():
    """
    Fills missing values in NUMERIC_COLUMNS with their column mean.
    Returns before/after missing counts and the fill values used.
    """
    df = load_data()

    # Record missing counts before imputation
    before = {col: int(df[col].isnull().sum()) for col in NUMERIC_COLUMNS}

    # Calculate means (on original data, before filling)
    fill_values = {col: round(float(df[col].mean()), 4) for col in NUMERIC_COLUMNS}

    # Apply mean fill
    mean_df = df.copy()
    mean_df[NUMERIC_COLUMNS] = mean_df[NUMERIC_COLUMNS].fillna(
        mean_df[NUMERIC_COLUMNS].mean()
    )

    # Record missing counts after imputation
    after = {col: int(mean_df[col].isnull().sum()) for col in NUMERIC_COLUMNS}

    return {
        "before": before,
        "after": after,
        "fill_values": fill_values,
        "fill_type": "Mean",
        "columns": NUMERIC_COLUMNS
    }


# ─────────────────────────────────────────
# Median Imputation
# ─────────────────────────────────────────

def median_imputation():
    """
    Fills missing values in NUMERIC_COLUMNS with their column median.
    Returns before/after missing counts and the fill values used.
    """
    df = load_data()

    before = {col: int(df[col].isnull().sum()) for col in NUMERIC_COLUMNS}

    fill_values = {col: round(float(df[col].median()), 4) for col in NUMERIC_COLUMNS}

    median_df = df.copy()
    median_df[NUMERIC_COLUMNS] = median_df[NUMERIC_COLUMNS].fillna(
        median_df[NUMERIC_COLUMNS].median()
    )

    after = {col: int(median_df[col].isnull().sum()) for col in NUMERIC_COLUMNS}

    return {
        "before": before,
        "after": after,
        "fill_values": fill_values,
        "fill_type": "Median",
        "columns": NUMERIC_COLUMNS
    }


# ─────────────────────────────────────────
# Mode Imputation
# ─────────────────────────────────────────

def mode_imputation():
    """
    Fills missing values in NUMERIC_COLUMNS with their column mode.
    Returns before/after missing counts and the fill values used.
    """
    df = load_data()

    before = {col: int(df[col].isnull().sum()) for col in NUMERIC_COLUMNS}

    fill_values = {col: round(float(df[col].mode()[0]), 4) for col in NUMERIC_COLUMNS}

    mode_df = df.copy()
    for col in NUMERIC_COLUMNS:
        mode_df[col] = mode_df[col].fillna(mode_df[col].mode()[0])

    after = {col: int(mode_df[col].isnull().sum()) for col in NUMERIC_COLUMNS}

    return {
        "before": before,
        "after": after,
        "fill_values": fill_values,
        "fill_type": "Mode",
        "columns": NUMERIC_COLUMNS
    }
