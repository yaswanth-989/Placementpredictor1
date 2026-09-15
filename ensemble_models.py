"""
ensemble_models.py
==================
Ensemble Methods module: Random Forest + AdaBoost.
Same data pipeline as decision_tree_module.py:
  - Target: PlacementStatus (classification)
  - Preprocessing: median imputation + pd.get_dummies
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


def _build_result(name, model, X_train, X_test, y_train, y_test, X_all):
    """Shared evaluation logic — returns the consistent result dict."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred = model.predict(X_test)
    train_acc = round(float(accuracy_score(y_train, model.predict(X_train))), 4)
    test_acc  = round(float(accuracy_score(y_test, y_pred)), 4)

    rpt     = classification_report(y_test, y_pred, output_dict=True)
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

    labels = [str(c) for c in sorted(y_test.unique())]
    cm     = confusion_matrix(y_test, y_pred).tolist()

    # Feature importances (top 10)
    imp = pd.Series(model.feature_importances_, index=X_all.columns)
    imp = imp.sort_values(ascending=False).head(10)
    feat_imp = [{"feature": col, "importance": round(float(v), 4)} for col, v in imp.items()]

    return {
        "method":        name,
        "train_size":    len(X_train),
        "test_size":     len(X_test),
        "train_acc":     train_acc,
        "test_acc":      test_acc,
        "n_features":    len(X_all.columns),
        "report_rows":   report_rows,
        "confusion_matrix":    cm,
        "cm_labels":           labels,
        "feature_importances": feat_imp,
        "weighted_f1":  round(avg.get("f1-score", 0), 4),
    }


# ─────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────

def random_forest():
    """
    RandomForestClassifier with 100 trees.
    Returns accuracy, classification report, confusion matrix, feature importances.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = _prepare_df()
    X  = _encode_X(df.drop(columns=[TARGET]))
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    result = _build_result(
        "Random Forest (100 trees)", model,
        X_train, X_test, y_train, y_test, X
    )
    result["params"] = {
        "n_estimators": 100,
        "max_depth":    "None (unlimited)",
        "criterion":    "gini",
        "bootstrap":    True,
        "n_jobs":       -1,
    }
    result["concept"] = (
        "Builds 100 independent Decision Trees on random subsets of data (bagging) "
        "and features. Final prediction = majority vote across all trees. "
        "Reduces variance without increasing bias."
    )
    return result


# ─────────────────────────────────────────
# AdaBoost
# ─────────────────────────────────────────

def adaboost():
    """
    AdaBoostClassifier with 50 estimators (Decision Tree stumps).
    Returns accuracy, classification report, confusion matrix, feature importances.
    """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    df = _prepare_df()
    X  = _encode_X(df.drop(columns=[TARGET]))
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(
        estimator=base,
        n_estimators=50,
        learning_rate=1.0,
        random_state=42,
        algorithm="SAMME",
    )
    model.fit(X_train, y_train)

    result = _build_result(
        "AdaBoost (50 estimators, Decision Stump)", model,
        X_train, X_test, y_train, y_test, X
    )
    result["params"] = {
        "n_estimators":  50,
        "base_estimator":"Decision Stump (max_depth=1)",
        "learning_rate": 1.0,
        "algorithm":     "SAMME",
    }
    result["concept"] = (
        "Trains weak learners sequentially. Each new learner focuses more on "
        "samples that previous learners misclassified (re-weighting). Final "
        "prediction is a weighted vote. Boosts weak learners into a strong classifier."
    )
    return result


# ─────────────────────────────────────────
# Ensemble Comparison (RF vs AdaBoost vs DT)
# ─────────────────────────────────────────

def ensemble_comparison():
    """
    Trains RF, AdaBoost and Decision Tree baseline, returns comparison table.
    """
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    df = _prepare_df()
    X  = _encode_X(df.drop(columns=[TARGET]))
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    configs = [
        ("Decision Tree (baseline)", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("Random Forest (100 trees)", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("AdaBoost (50 estimators)",  AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=50, random_state=42, algorithm="SAMME"
        )),
    ]

    rows = []
    for name, model in configs:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tr = round(float(accuracy_score(y_train, model.predict(X_train))), 4)
        te = round(float(accuracy_score(y_test,  y_pred)),                 4)
        rpt = classification_report(y_test, y_pred, output_dict=True)
        f1  = round(rpt.get("weighted avg", {}).get("f1-score", 0), 4)
        gap = round(tr - te, 4)
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
        "method":        "Ensemble Methods Comparison",
        "models":        rows,
        "best_model":    best["model"],
        "best_test_acc": best["test_acc"],
    }
