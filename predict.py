"""
predict.py
==========
Prediction module — trains the best model on the full dataset (cached),
then predicts PlacementStatus for a new student's input.

Supports 6 models. Model is trained once per server session and cached.
"""

import numpy as np
import pandas as pd
from load_data import load_data

TARGET       = "PlacementStatus"
COLS_TO_DROP = ["StudentID", "Salary Package", "IsAnomaly", "CGPA_Tier", TARGET]

# ── numeric features shown in the form ──────────────────────────────
NUMERIC_FEATURES = [
    "CGPA",
    "AttendancePercent",
    "Internships", "Projects", "Workshops", "Certifications",
    "Publications", "AptitudeTestScore", "SoftSkillsRating",
    "CodingTestScore", "MockInterviewScore", "ExtraCurricular",
]

# ── all SGPA columns (filled automatically from CGPA) ────────────────
SGPA_COLS = [f"SGPA_Sem{i}" for i in range(1, 9)]

# ── categorical features shown in the form ──────────────────
CATEGORICAL_FEATURES = {
    "Gender":            ["Male", "Female"],
    "CollegeTier":       ["Tier1", "Tier2", "Tier3"],
    "Stream":            ["CSE", "ECE", "EEE", "Mechanical", "Civil", "IT", "Other"],
    "Specialisation":    ["Data Science", "Networking", "Web Dev", "Embedded", "AI/ML", "Other"],
    "Hostel":            ["Yes", "No"],
    "HistoryOfBacklogs": ["Yes", "No"],
}

# Defaults shown in form
NUMERIC_DEFAULTS = {
    "CGPA": 7.5, "AttendancePercent": 80.0,
    "Internships": 1, "Projects": 2, "Workshops": 1, "Certifications": 2,
    "Publications": 0, "AptitudeTestScore": 65.0, "SoftSkillsRating": 3.5,
    "CodingTestScore": 60.0, "MockInterviewScore": 60.0, "ExtraCurricular": 1,
    # SGPAs auto-filled from CGPA — not shown in form but kept for reference
    **{f"SGPA_Sem{i}": 7.5 for i in range(1, 9)},
}

# Groups for the form layout
FORM_GROUPS = [
    ("🎓 Academic Performance",
     ["CGPA", "AttendancePercent"]),
    ("🎯 Test Scores",
     ["AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "SoftSkillsRating"]),
    ("🚀 Activities & Experience",
     ["Internships", "Projects", "Certifications", "Workshops",
      "Publications", "ExtraCurricular"]),
]

# ── In-memory model cache (trained once per session) ─────────
_cache = {}   # { model_name: (model, feature_columns, label_encoder) }


def _prepare_training_data():
    """Load, clean and encode full dataset for model training."""
    from sklearn.impute import SimpleImputer
    df = load_data()
    df = df.drop(columns=[c for c in COLS_TO_DROP + ["City"] if c in df.columns])
    df = df.dropna(subset=[TARGET])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Impute numeric
    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = list(X.select_dtypes(exclude="number").columns)

    if num_cols:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y


def _train_and_cache(model_name: str):
    """Train the requested model on full data and store in cache."""
    from sklearn.preprocessing import LabelEncoder

    X, y = _prepare_training_data()
    feature_cols = list(X.columns)

    # Encode y (needed for XGBoost / LightGBM)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if model_name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)
        _cache[model_name] = (model, feature_cols, None)

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        _cache[model_name] = (model, feature_cols, None)

    elif model_name == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=50, random_state=42, algorithm="SAMME"
        )
        model.fit(X, y)
        _cache[model_name] = (model, feature_cols, None)

    elif model_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        model.fit(X, y)
        _cache[model_name] = (model, feature_cols, None)

    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            eval_metric="logloss", random_state=42, verbosity=0
        )
        model.fit(X, y_enc)
        _cache[model_name] = (model, feature_cols, le)

    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=100, learning_rate=0.1, num_leaves=31,
            random_state=42, verbosity=-1
        )
        model.fit(X, y_enc)
        _cache[model_name] = (model, feature_cols, le)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def _build_input_row(form_data: dict, feature_cols: list) -> pd.DataFrame:
    """
    Convert flat form dict → a 1-row DataFrame aligned to training feature_cols.
    Handles one-hot encoded columns by matching prefixes.
    """
    row = {}

    # Numeric fields (user-visible ones)
    for feat in NUMERIC_FEATURES:
        try:
            row[feat] = float(form_data.get(feat, NUMERIC_DEFAULTS.get(feat, 0)))
        except (ValueError, TypeError):
            row[feat] = NUMERIC_DEFAULTS.get(feat, 0)

    # Auto-fill all SGPA columns from CGPA (not shown in form)
    cgpa_val = row.get("CGPA", 7.5)
    for sgpa_col in SGPA_COLS:
        row[sgpa_col] = cgpa_val

    # Categorical fields → one-hot (same get_dummies logic used in training)
    cat_input = {}
    for feat, options in CATEGORICAL_FEATURES.items():
        cat_input[feat] = form_data.get(feat, options[0])

    cat_df = pd.DataFrame([cat_input])
    cat_dummies = pd.get_dummies(cat_df, drop_first=True)

    # Merge numeric + dummies
    num_df = pd.DataFrame([row])
    combined = pd.concat([num_df, cat_dummies], axis=1)

    # Align to training columns (add missing cols as 0, drop extra)
    for col in feature_cols:
        if col not in combined.columns:
            combined[col] = 0
    combined = combined[feature_cols]

    return combined


def predict_placement(form_data: dict, model_name: str = "random_forest") -> dict:
    """
    Main prediction entry point.
    Returns: {
        prediction: "Placed" | "Not Placed",
        probability: float (0–100 %),
        confidence: "High" | "Medium" | "Low",
        model_used: str,
        top_features: [{feature, importance}],
        cached: bool,
    }
    """
    # Train and cache if not already done
    was_cached = model_name in _cache
    if not was_cached:
        _train_and_cache(model_name)

    model, feature_cols, le = _cache[model_name]

    # Build input row
    X_input = _build_input_row(form_data, feature_cols)

    # Predict
    raw_pred = model.predict(X_input)[0]

    # Decode prediction label
    if le is not None:
        label = le.inverse_transform([raw_pred])[0]
    else:
        label = raw_pred

    # Probability
    prob_arr = None
    try:
        prob_arr = model.predict_proba(X_input)[0]
        # prob_arr index 0 → first class, index 1 → second class
        classes = le.classes_ if le else model.classes_
        class_idx = list(classes).index(raw_pred)
        prob = float(prob_arr[class_idx]) * 100
    except Exception:
        prob = None

    # Map to string
    if label == 1 or label == "1":
        label_str = "Placed"
    elif label == 0 or label == "0":
        label_str = "Not Placed"
    else:
        label_str = str(label)

    # Confidence tier
    if prob is not None:
        confidence = "High" if prob >= 80 else "Medium" if prob >= 60 else "Low"
    else:
        confidence = "—"

    # Top 10 feature importances
    top_features = []
    try:
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=False).head(10)
        for feat, val in imp.items():
            top_features.append({"feature": feat, "importance": round(float(val), 4)})
    except Exception:
        pass

    return {
        "prediction":  label_str,
        "probability": round(prob, 1) if prob is not None else None,
        "confidence":  confidence,
        "model_used":  model_name,
        "top_features": top_features,
        "cached":      was_cached,
        "placed":      label_str == "Placed",
    }


# ── Helpers for the template ─────────────────────────────────

def get_form_meta() -> dict:
    """Return all metadata the template needs to render the input form."""
    return {
        "numeric_features":     NUMERIC_FEATURES,
        "numeric_defaults":     NUMERIC_DEFAULTS,
        "categorical_features": CATEGORICAL_FEATURES,
        "form_groups":          FORM_GROUPS,
        "models": [
            {"key": "random_forest",     "label": "🌲 Random Forest",      "badge": "badge-rf"},
            {"key": "lightgbm",          "label": "💡 LightGBM",            "badge": "badge-lgbm"},
            {"key": "xgboost",           "label": "⚡ XGBoost",             "badge": "badge-xgb"},
            {"key": "gradient_boosting", "label": "📉 Gradient Boosting",   "badge": "badge-gb"},
            {"key": "adaboost",          "label": "🔁 AdaBoost",            "badge": "badge-ada"},
            {"key": "decision_tree",     "label": "🌿 Decision Tree",       "badge": "badge-dt"},
        ],
    }
