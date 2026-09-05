"""
linear_regression.py
====================
Linear Regression module — covers all 5 reference programs:
  1. Simple LR (CGPA → Salary Package)
  2. Multiple LR (all features → Salary, with Pipeline / no data leakage)
  3. Data Leakage Demo (leaked vs correct workflow comparison)
"""

import numpy as np
import pandas as pd
from load_data import load_data

TARGET       = "Salary Package"
COLS_TO_DROP = ["StudentID", "IsAnomaly"]


def _prepare_df():
    """Load data, drop ID/leakage columns, remove rows missing the target."""
    df = load_data()
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    df = df.dropna(subset=[TARGET])
    return df


# ─────────────────────────────────────────
# Overview — Correlation with Salary Package
# ─────────────────────────────────────────

def get_lr_overview():
    """Correlation of each numeric feature with Salary Package."""
    df = _prepare_df()
    num_df = df.select_dtypes(include="number")
    if TARGET not in num_df.columns:
        return {}

    corr = num_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    corr_rows = []
    for col, val in corr.items():
        corr_rows.append({
            "feature":   col,
            "correlation": round(float(val), 4),
            "strength":  "Strong"   if abs(val) > 0.5 else
                         "Moderate" if abs(val) > 0.3 else "Weak",
            "direction": "Positive" if val >= 0 else "Negative",
        })

    s = df[TARGET]
    return {
        "target":      TARGET,
        "n_records":   len(df),
        "corr_rows":   corr_rows,
        "target_stats": {
            "Mean":   round(float(s.mean()),   2),
            "Median": round(float(s.median()), 2),
            "Std":    round(float(s.std()),    2),
            "Min":    round(float(s.min()),    2),
            "Max":    round(float(s.max()),    2),
        },
    }


# ─────────────────────────────────────────
# Simple Linear Regression  (CGPA → Salary)
# ─────────────────────────────────────────

def simple_linear_regression():
    """
    Single-feature LR: CGPA → Salary Package.
    Pipeline: SimpleImputer → LinearRegression.
    Returns formula, metrics (MAE / MSE / RMSE / R²), and a 10-row comparison table.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    df = _prepare_df()
    FEATURE = "CGPA"
    if FEATURE not in df.columns:
        return {"error": f"'{FEATURE}' column not found"}

    X = df[[FEATURE]]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    coef      = float(pipe.named_steps["model"].coef_[0])
    intercept = float(pipe.named_steps["model"].intercept_)
    mae  = float(mean_absolute_error(y_test, y_pred))
    mse  = float(mean_squared_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    comparison = []
    for actual, pred in list(zip(y_test, y_pred))[:10]:
        comparison.append({
            "actual":    round(float(actual), 2),
            "predicted": round(float(pred),   2),
            "error":     round(abs(float(actual) - float(pred)), 2),
        })

    return {
        "method":      "Simple Linear Regression",
        "feature":     FEATURE,
        "target":      TARGET,
        "formula":     f"Salary = {round(coef,4)} × CGPA + {round(intercept,4)}",
        "coefficient": round(coef,      4),
        "intercept":   round(intercept, 4),
        "train_size":  len(X_train),
        "test_size":   len(X_test),
        "metrics": {
            "MAE":  round(mae,               2),
            "MSE":  round(mse,               2),
            "RMSE": round(float(np.sqrt(mse)), 2),
            "R²":   round(r2,                4),
        },
        "comparison": comparison,
    }


# ─────────────────────────────────────────
# Multiple Linear Regression (Pipeline — no leakage)
# ─────────────────────────────────────────

def multiple_linear_regression():
    """
    Multiple LR: all features → Salary Package.
    ColumnTransformer + Pipeline (split first, then preprocess).
    Returns metrics, top numeric coefficients, and 10-row comparison.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    df = _prepare_df()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = list(X.select_dtypes(exclude="number").columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    full_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        LinearRegression()),
    ])
    full_pipe.fit(X_train, y_train)
    y_pred = full_pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    r2  = float(r2_score(y_test, y_pred))

    model  = full_pipe.named_steps["model"]
    coefs  = [
        {"feature": col, "coefficient": round(float(model.coef_[i]), 4)}
        for i, col in enumerate(num_cols)
    ]
    coefs.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    comparison = []
    for actual, pred in list(zip(y_test, y_pred))[:10]:
        comparison.append({
            "actual":    round(float(actual), 2),
            "predicted": round(float(pred),   2),
            "error":     round(abs(float(actual) - float(pred)), 2),
        })

    return {
        "method":         "Multiple Linear Regression (Pipeline — No Data Leakage)",
        "num_features":   len(num_cols),
        "cat_features":   len(cat_cols),
        "total_features": len(X.columns),
        "target":         TARGET,
        "train_size":     len(X_train),
        "test_size":      len(X_test),
        "metrics": {
            "MAE":  round(mae,                 2),
            "MSE":  round(mse,                 2),
            "RMSE": round(float(np.sqrt(mse)), 2),
            "R²":   round(r2,                  4),
        },
        "top_coefficients": coefs[:10],
        "comparison":       comparison,
    }


# ─────────────────────────────────────────
# Data Leakage Demo
# ─────────────────────────────────────────

def data_leakage_demo():
    """
    Leaked workflow: global impute + encode BEFORE split → inflated metrics.
    Correct workflow: split FIRST, preprocess INSIDE pipeline.
    Returns both metric sets and the R² gap.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    df = _prepare_df()
    X_raw = df.drop(columns=[TARGET])
    y     = df[TARGET]

    num_cols = list(X_raw.select_dtypes(include="number").columns)
    cat_cols = list(X_raw.select_dtypes(exclude="number").columns)

    # ── LEAKED WORKFLOW ──────────────────────────────────────────
    X_leaked = X_raw.copy()
    # Global imputation (includes test data → LEAKAGE)
    for col in num_cols:
        X_leaked[col] = X_leaked[col].fillna(X_leaked[col].median())
    # Global label encoding (LEAKAGE)
    for col in cat_cols:
        X_leaked[col] = LabelEncoder().fit_transform(X_leaked[col].astype(str))

    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(
        X_leaked, y, test_size=0.2, random_state=42
    )
    leaked_model = LinearRegression().fit(Xl_tr, yl_tr)
    ypl = leaked_model.predict(Xl_te)

    leaked_metrics = {
        "MAE":  round(float(mean_absolute_error(yl_te, ypl)),           2),
        "RMSE": round(float(np.sqrt(mean_squared_error(yl_te, ypl))),   2),
        "R²":   round(float(r2_score(yl_te, ypl)),                      4),
    }

    # ── CORRECT WORKFLOW ─────────────────────────────────────────
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    correct_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        LinearRegression()),
    ])
    correct_pipe.fit(Xc_tr, yc_tr)
    ypc = correct_pipe.predict(Xc_te)

    correct_metrics = {
        "MAE":  round(float(mean_absolute_error(yc_te, ypc)),           2),
        "RMSE": round(float(np.sqrt(mean_squared_error(yc_te, ypc))),   2),
        "R²":   round(float(r2_score(yc_te, ypc)),                      4),
    }

    return {
        "method":       "Data Leakage: Leaked vs Correct Workflow",
        "leaked":       leaked_metrics,
        "correct":      correct_metrics,
        "r2_diff":      round(abs(leaked_metrics["R²"] - correct_metrics["R²"]), 4),
        "mae_diff":     round(abs(leaked_metrics["MAE"] - correct_metrics["MAE"]), 2),
        "steps": [
            {
                "step":    "1. Global Imputation",
                "leaked":  "Computed on FULL dataset (test stats leak in)",
                "correct": "Fit on TRAINING set only, applied to test",
            },
            {
                "step":    "2. Categorical Encoding",
                "leaked":  "LabelEncoder on full dataset (LEAKAGE)",
                "correct": "OneHotEncoder fit on train inside Pipeline",
            },
            {
                "step":    "3. Train / Test Split",
                "leaked":  "AFTER preprocessing (wrong order)",
                "correct": "BEFORE preprocessing (correct order)",
            },
        ],
    }
