"""
decision_tree_module.py
=======================
Decision Tree module — based on decision__tree_1.py reference program.
Covers Entropy, Gini, Information Gain, model training (criterion=gini,
entropy), confusion matrix, feature importance, and overfitting/pruning
comparison (deep tree / max_depth=5 / min_samples_leaf=20 / ccp_alpha).
"""

import numpy as np
import pandas as pd
from load_data import load_data

TARGET       = "PlacementStatus"
COLS_TO_DROP = ["StudentID", "Salary Package", "IsAnomaly"]


def _prepare_df():
    """Load and prepare classification data."""
    df = load_data()
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    df = df.dropna(subset=[TARGET])
    return df


def _encode_X(X):
    """Impute numeric NaN with median, one-hot encode categorical columns."""
    from sklearn.impute import SimpleImputer

    X = X.copy()   # avoid SettingWithCopyWarning
    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = list(X.select_dtypes(exclude="number").columns)

    if num_cols:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X


# ─────────────────────────────────────────
# Entropy & Gini Demo
# ─────────────────────────────────────────

def entropy_gini_demo():
    """
    Demonstrates Entropy, Gini impurity, and Information Gain using
    the real placement data (PlacementStatus column with CGPA splits).
    """
    df = load_data()
    df = df.dropna(subset=[TARGET, "CGPA"])
    y  = df[TARGET]
    n  = len(y)

    vc = y.value_counts(normalize=True)

    # Overall entropy
    entropy_val = float(-sum(p * np.log2(p + 1e-15) for p in vc))
    # Overall Gini
    gini_val = float(1 - sum(p ** 2 for p in vc))

    # Class distribution
    class_dist = {str(k): int(v) for k, v in y.value_counts().items()}

    # Helper: entropy of a Series
    def _entropy(s):
        p_ = s.value_counts(normalize=True)
        return float(-sum(p * np.log2(p + 1e-15) for p in p_))

    # Information Gain for CGPA split thresholds
    thresholds = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
    threshold_results = []
    for t in thresholds:
        left  = y[df["CGPA"] <= t]
        right = y[df["CGPA"] >  t]
        if len(left) == 0 or len(right) == 0:
            continue
        child_entropy = (len(left) / n) * _entropy(left) + (len(right) / n) * _entropy(right)
        ig = float(entropy_val - child_entropy)

        left_vc  = left.value_counts().to_dict()
        right_vc = right.value_counts().to_dict()
        placed_left  = int(left_vc.get(1,  left_vc.get("Placed", 0)))
        placed_right = int(right_vc.get(1, right_vc.get("Placed", 0)))

        threshold_results.append({
            "threshold":        t,
            "left_count":       len(left),
            "right_count":      len(right),
            "ig":               round(ig, 4),
            "placed_left_pct":  round((placed_left  / len(left))  * 100, 1),
            "placed_right_pct": round((placed_right / len(right)) * 100, 1),
        })

    best = max(threshold_results, key=lambda x: x["ig"]) if threshold_results else None

    return {
        "n_records":      n,
        "class_dist":     class_dist,
        "entropy":        round(entropy_val, 4),
        "gini":           round(gini_val,    4),
        "thresholds":     threshold_results,
        "best_threshold": best,
        "formulas": {
            "Entropy":         "H(S) = −Σ p_i × log₂(p_i)",
            "Gini":            "Gini(S) = 1 − Σ p_i²",
            "Information Gain":"IG(S,A) = H(S) − Σ (|Sv|/|S|) × H(Sv)",
        },
        "concept_table": [
            {"concept": "Entropy",         "formula": "−Σ p log₂p",    "goal": "Measure uncertainty"},
            {"concept": "Information Gain","formula": "Parent H − Child H","goal": "Maximise (best split)"},
            {"concept": "Gini",            "formula": "1 − Σ p²",       "goal": "Measure impurity"},
            {"concept": "Gini Decrease",   "formula": "Parent Gini − Child Gini","goal": "Maximise"},
        ],
    }


# ─────────────────────────────────────────
# Train Decision Tree
# ─────────────────────────────────────────

def train_decision_tree():
    """
    Trains DecisionTreeClassifier (criterion=gini, max_depth=5) — same as reference.
    Returns accuracy, classification report, confusion matrix, feature importances.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    df = _prepare_df()
    X  = _encode_X(df.drop(columns=[TARGET]))
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_acc = round(float(accuracy_score(y_train, model.predict(X_train))), 4)
    test_acc  = round(float(accuracy_score(y_test,  y_pred)),                 4)

    # Classification report
    rpt = classification_report(y_test, y_pred, output_dict=True)
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

    # Confusion matrix
    labels = [str(c) for c in sorted(y.unique())]
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature importance (top 10)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False).head(10)
    feature_importances = [
        {"feature": col, "importance": round(float(v), 4)}
        for col, v in imp.items()
    ]

    return {
        "method":       "Decision Tree (criterion=gini, max_depth=5)",
        "train_size":   len(X_train),
        "test_size":    len(X_test),
        "train_acc":    train_acc,
        "test_acc":     test_acc,
        "depth":        model.get_depth(),
        "n_leaves":     model.get_n_leaves(),
        "n_features":   len(X.columns),
        "report_rows":  report_rows,
        "confusion_matrix": cm,
        "cm_labels":    labels,
        "feature_importances": feature_importances,
    }


# ─────────────────────────────────────────
# Model Comparison — Overfitting Analysis
# ─────────────────────────────────────────

def model_comparison():
    """
    Trains 4 Decision Tree variants to show the overfitting / pruning trade-off:
      1. Deep Tree (unconstrained)
      2. max_depth = 5
      3. min_samples_leaf = 20
      4. ccp_alpha tuned (Cost-Complexity Pruning)
    Returns comparison table with train/test accuracy, depth, leaves, verdict.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from typing import Any, Dict, List

    df = _prepare_df()
    X  = _encode_X(df.drop(columns=[TARGET]))
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Estimate best ccp_alpha on a 5 K-row sample (fast)
    sample_n = min(5000, len(X_train))
    path     = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(
        X_train.iloc[:sample_n], y_train.iloc[:sample_n]
    )
    alphas      = path.ccp_alphas
    best_alpha  = float(alphas[max(1, len(alphas) // 3)]) if len(alphas) > 1 else 0.001

    configs: List[Dict[str, Any]] = [
        {"name": "Deep Tree (Overfitting)",
         "params": {"random_state": 42}},
        {"name": "max_depth = 5",
         "params": {"max_depth": 5, "random_state": 42}},
        {"name": "min_samples_leaf = 20",
         "params": {"min_samples_leaf": 20, "random_state": 42}},
        {"name": f"CCP Pruning (α = {round(best_alpha, 5)})",
         "params": {"ccp_alpha": best_alpha, "random_state": 42}},
    ]

    results = []
    for cfg in configs:
        m = DecisionTreeClassifier(**cfg["params"])
        m.fit(X_train, y_train)
        tr = round(float(accuracy_score(y_train, m.predict(X_train))), 4)
        te = round(float(accuracy_score(y_test,  m.predict(X_test))),  4)
        gap = round(tr - te, 4)
        results.append({
            "model":       cfg["name"],
            "train_acc":   tr,
            "test_acc":    te,
            "overfit_gap": gap,
            "depth":       m.get_depth(),
            "n_leaves":    m.get_n_leaves(),
            "verdict":     "⚠️ Overfitting" if gap > 0.05 else "✅ Good Fit",
        })

    best = max(results, key=lambda x: x["test_acc"])

    return {
        "method":        "Decision Tree — Overfitting & Pruning Analysis",
        "models":        results,
        "best_model":    best["model"],
        "best_test_acc": best["test_acc"],
    }
