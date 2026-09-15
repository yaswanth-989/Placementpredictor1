"""
boosting_models.py
==================
Boosting Methods module: Gradient Boosting (sklearn) + XGBoost + LightGBM.
Same data pipeline as ensemble_models.py.
"""

import numpy as np
import pandas as pd
from load_data import load_data

TARGET       = "PlacementStatus"
COLS_TO_DROP = ["StudentID", "Salary Package", "IsAnomaly"]


def _prepare_df():
    df = load_data()
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    df = df.dropna(subset=[TARGET])
    return df


def _encode_X(X):
    from sklearn.impute import SimpleImputer
    X = X.copy()
    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = list(X.select_dtypes(exclude="number").columns)
    if num_cols:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X


def _encode_y(y):
    """Encode string labels to 0/1 for XGBoost compatibility."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    return le.fit_transform(y), le


def _build_result(name, model, X_train, X_test, y_train, y_test, X_all,
                  le=None):
    """Shared evaluation — returns consistent result dict."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred_raw = model.predict(X_test)
    y_tr_raw   = model.predict(X_train)

    # Decode labels if they were encoded
    if le is not None:
        y_pred   = le.inverse_transform(y_pred_raw)
        y_tr_dec = le.inverse_transform(y_tr_raw)
        y_test_d = le.inverse_transform(y_test)
        y_train_d= le.inverse_transform(y_train)
    else:
        y_pred    = y_pred_raw
        y_tr_dec  = y_tr_raw
        y_test_d  = y_test
        y_train_d = y_train

    train_acc = round(float(accuracy_score(y_train_d, y_tr_dec)), 4)
    test_acc  = round(float(accuracy_score(y_test_d,  y_pred)),   4)

    rpt     = classification_report(y_test_d, y_pred, output_dict=True)
    classes = sorted([k for k in rpt if k not in ("accuracy", "macro avg", "weighted avg")])
    report_rows = []
    for cls in classes:
        r = rpt[cls]
        report_rows.append({
            "class":     str(cls),
            "precision": round(r["precision"], 4),
            "recall":    round(r["recall"],    4),
            "f1_score":  round(r["f1-score"],  4),
            "support":   int(r["support"]),
        })
    avg = rpt.get("weighted avg", {})
    report_rows.append({
        "class":     "weighted avg",
        "precision": round(avg.get("precision", 0), 4),
        "recall":    round(avg.get("recall",    0), 4),
        "f1_score":  round(avg.get("f1-score",  0), 4),
        "support":   int(avg.get("support",     0)),
    })

    labels = [str(c) for c in sorted(pd.Series(y_test_d).unique())]
    cm     = confusion_matrix(y_test_d, y_pred, labels=sorted(pd.Series(y_test_d).unique())).tolist()

    # Feature importances (top 10)
    imp = pd.Series(model.feature_importances_, index=X_all.columns)
    imp = imp.sort_values(ascending=False).head(10)
    feat_imp = [{"feature": col, "importance": round(float(v), 4)} for col, v in imp.items()]

    return {
        "method":              name,
        "train_size":          len(X_train),
        "test_size":           len(X_test),
        "train_acc":           train_acc,
        "test_acc":            test_acc,
        "n_features":          len(X_all.columns),
        "report_rows":         report_rows,
        "confusion_matrix":    cm,
        "cm_labels":           labels,
        "feature_importances": feat_imp,
        "weighted_f1":         round(avg.get("f1-score", 0), 4),
    }


# ─────────────────────────────────────────
# Gradient Boosting (sklearn)
# ─────────────────────────────────────────

def gradient_boosting():
    """
    GradientBoostingClassifier: 100 estimators, learning_rate=0.1, max_depth=3.
    sklearn's classic sequential boosting implementation.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    df = _prepare_df()
    X  = _encode_X(df.drop(columns=[TARGET]))
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    result = _build_result(
        "Gradient Boosting (sklearn, 100 estimators)", model,
        X_train, X_test, y_train, y_test, X
    )
    result["params"] = {
        "n_estimators":  100,
        "learning_rate": 0.1,
        "max_depth":     3,
        "subsample":     0.8,
        "loss":          "log_loss",
    }
    result["concept"] = (
        "Builds trees sequentially — each tree corrects errors of the previous one "
        "by fitting on the negative gradient of the loss. Slower but often very accurate. "
        "sklearn's original implementation."
    )
    return result


# ─────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────

def xgboost_model():
    """
    XGBClassifier: 100 estimators, learning_rate=0.1.
    Regularised gradient boosting with parallel tree building.
    """
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    df = _prepare_df()
    X  = _encode_X(df.drop(columns=[TARGET]))
    y  = df[TARGET]

    # Encode string labels → 0/1 (XGBoost requirement)
    y_enc, le = _encode_y(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    result = _build_result(
        "XGBoost (100 estimators)", model,
        X_train, X_test, y_train, y_test, X, le=le
    )
    result["params"] = {
        "n_estimators":    100,
        "learning_rate":   0.1,
        "max_depth":       6,
        "subsample":       0.8,
        "colsample_bytree":0.8,
        "eval_metric":     "logloss",
    }
    result["concept"] = (
        "Extreme Gradient Boosting — an optimised, regularised implementation of "
        "gradient boosting with built-in L1/L2 regularisation, parallel tree "
        "construction, and missing value handling. Extremely fast and accurate."
    )
    return result


# ─────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────

def lightgbm_model():
    """
    LGBMClassifier: leaf-wise tree growth for speed on large datasets.
    """
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split

    df = _prepare_df()
    X  = _encode_X(df.drop(columns=[TARGET]))
    y  = df[TARGET]

    # Encode labels for LightGBM
    y_enc, le = _encode_y(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,          # no limit — controlled by num_leaves
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,           # silence output
    )
    model.fit(X_train, y_train)

    result = _build_result(
        "LightGBM (100 estimators)", model,
        X_train, X_test, y_train, y_test, X, le=le
    )
    result["params"] = {
        "n_estimators":  100,
        "learning_rate": 0.1,
        "num_leaves":    31,
        "subsample":     0.8,
        "colsample_bytree": 0.8,
        "tree_method":   "leaf-wise (faster)",
    }
    result["concept"] = (
        "Microsoft's Light Gradient Boosting Machine grows trees leaf-wise (best-first) "
        "instead of level-wise, finding the leaf with the max delta-loss. Extremely fast "
        "on large datasets. Uses histogram binning for further speed gains."
    )
    return result


# ─────────────────────────────────────────
# Boosting Comparison (GB + XGB + LGBM)
# ─────────────────────────────────────────

def boosting_comparison():
    """
    Trains all 3 boosting models + DT baseline and returns comparison table.
    """
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    df = _prepare_df()
    X  = _encode_X(df.drop(columns=[TARGET]))
    y  = df[TARGET]

    # Numeric labels for XGB + LGBM
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)

    # Splits (stratified on string labels for DT/GB, numeric for XGB/LGBM)
    X_tr, X_te, y_tr_s, y_te_s = train_test_split(X, y,     test_size=0.2, random_state=42, stratify=y)
    _,    _,    y_tr_n, y_te_n  = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    configs = [
        ("Decision Tree (baseline)",
         DecisionTreeClassifier(max_depth=5, random_state=42),
         y_tr_s, y_te_s, None),

        ("Gradient Boosting (sklearn)",
         GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
         y_tr_s, y_te_s, None),

        ("XGBoost",
         XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                       eval_metric="logloss", random_state=42, verbosity=0),
         y_tr_n, y_te_n, le),

        ("LightGBM",
         LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                        random_state=42, verbosity=-1),
         y_tr_n, y_te_n, le),
    ]

    rows = []
    for name, model, y_tr, y_te, enc in configs:
        model.fit(X_tr, y_tr)
        y_pred_raw = model.predict(X_te)
        y_te_d     = enc.inverse_transform(y_te)     if enc else y_te
        y_pred_d   = enc.inverse_transform(y_pred_raw) if enc else y_pred_raw
        y_tr_d     = enc.inverse_transform(y_tr)     if enc else y_tr
        y_trp_d    = enc.inverse_transform(model.predict(X_tr)) if enc else model.predict(X_tr)

        tr = round(float(accuracy_score(y_tr_d, y_trp_d)), 4)
        te = round(float(accuracy_score(y_te_d, y_pred_d)), 4)
        rpt= classification_report(y_te_d, y_pred_d, output_dict=True)
        f1 = round(rpt.get("weighted avg", {}).get("f1-score", 0), 4)
        gap= round(tr - te, 4)
        rows.append({
            "model":       name,
            "train_acc":   tr,
            "test_acc":    te,
            "f1":          f1,
            "overfit_gap": gap,
            "verdict":     "⚠️ Overfitting" if gap > 0.05 else "✅ Good Fit",
        })

    best = max(rows, key=lambda x: x["test_acc"])
    return {
        "method":        "Boosting Models Comparison",
        "models":        rows,
        "best_model":    best["model"],
        "best_test_acc": best["test_acc"],
    }
