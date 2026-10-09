"""
model_evaluation.py
===================
Model Performance Analysis & Hyperparameter Optimization Module:
1. Cross-Validation (k-Fold stratified) & 3-Way Splits (Train / Val / Test).
2. Classification Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix.
3. Calibration Curves (Reliability Diagram) & Brier Scores.
4. Regression Metrics: RMSE, MAE, R², MAPE, Residuals vs Predicted for Salary Package.
5. Hyperparameter Search: Grid Search vs Random Search vs Bayesian Optimization (Optuna TPE).
6. Comprehensive Evaluation Scorecard & Synthesis.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import optuna

from load_data import load_data

# Suppress verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# In-memory cache for fast sub-second UI interactions
_EVAL_CACHE = {}


def _get_dataset(sample_size=3000, random_state=42):
    """
    Prepares features X, classification target y_cls (PlacementStatus: 0/1),
    and regression target y_reg (Salary Package).
    """
    cache_key = f"eval_data_{sample_size}_{random_state}"
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    df_full = load_data()
    if len(df_full) > sample_size:
        df = df_full.sample(sample_size, random_state=random_state).copy()
    else:
        df = df_full.copy()

    # Targets
    y_cls = df["PlacementStatus"].astype(int).values if "PlacementStatus" in df else np.zeros(len(df))
    y_reg = df["Salary Package"].fillna(df["Salary Package"].median()).values if "Salary Package" in df else np.zeros(len(df))

    # Feature matrix (drop IDs and targets)
    drop_cols = ["StudentID", "PlacementStatus", "Salary Package", "IsAnomaly"]
    features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    num_cols = list(features_df.select_dtypes(include="number").columns)
    cat_cols = list(features_df.select_dtypes(exclude="number").columns)

    if num_cols:
        imputer = SimpleImputer(strategy="median")
        features_df[num_cols] = imputer.fit_transform(features_df[num_cols])

    if cat_cols:
        encoded_df = pd.get_dummies(features_df, columns=cat_cols, drop_first=True)
    else:
        encoded_df = features_df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded_df)

    data = {
        "X": X_scaled,
        "y_cls": y_cls,
        "y_reg": y_reg,
        "feature_names": list(encoded_df.columns),
        "n_samples": len(df),
        "n_features": X_scaled.shape[1]
    }
    _EVAL_CACHE[cache_key] = data
    return data


# ─────────────────────────────────────────────────────────
# 1. DATA SPLITS & CROSS-VALIDATION
# ─────────────────────────────────────────────────────────

def evaluate_splits(cv_folds=5):
    """
    Evaluates:
    1. 3-Way Split (Train 70%, Val 15%, Test 15%) with Generalization Gap.
    2. k-Fold Stratified Cross-Validation with per-fold metrics & confidence intervals.
    """
    cv_folds = max(3, min(int(cv_folds), 10))
    cache_key = f"splits_cv{cv_folds}"
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    t0 = time.time()
    data = _get_dataset()
    X = data["X"]
    y = data["y_cls"]

    # 1. 3-Way Split: 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    model_3way = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    model_3way.fit(X_train, y_train)

    train_acc = round(float(accuracy_score(y_train, model_3way.predict(X_train))), 4)
    val_acc = round(float(accuracy_score(y_val, model_3way.predict(X_val))), 4)
    test_acc = round(float(accuracy_score(y_test, model_3way.predict(X_test))), 4)
    gen_gap = round(float(train_acc - test_acc), 4)

    three_way_split = {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "train_pct": 70,
        "val_pct": 15,
        "test_pct": 15,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "generalization_gap": gen_gap,
        "overfitting_risk": "Low (< 3%)" if gen_gap < 0.03 else ("Moderate" if gen_gap < 0.08 else "High")
    }

    # 2. k-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    cv_res = cross_validate(cv_model, X, y, cv=skf, scoring=scoring, return_train_score=True)

    fold_metrics = []
    for i in range(cv_folds):
        fold_metrics.append({
            "fold": i + 1,
            "train_acc": round(float(cv_res["train_accuracy"][i]), 4),
            "test_acc": round(float(cv_res["test_accuracy"][i]), 4),
            "precision": round(float(cv_res["test_precision"][i]), 4),
            "recall": round(float(cv_res["test_recall"][i]), 4),
            "f1": round(float(cv_res["test_f1"][i]), 4),
            "roc_auc": round(float(cv_res["test_roc_auc"][i]), 4)
        })

    mean_acc = round(float(np.mean(cv_res["test_accuracy"])), 4)
    std_acc = round(float(np.std(cv_res["test_accuracy"])), 4)
    mean_f1 = round(float(np.mean(cv_res["test_f1"])), 4)
    std_f1 = round(float(np.std(cv_res["test_f1"])), 4)
    mean_auc = round(float(np.mean(cv_res["test_roc_auc"])), 4)
    std_auc = round(float(np.std(cv_res["test_roc_auc"])), 4)

    cv_summary = {
        "n_folds": cv_folds,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci_95": f"[{round(mean_acc - 1.96*std_acc, 4)} , {round(mean_acc + 1.96*std_acc, 4)}]",
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "folds": fold_metrics
    }

    elapsed = round(time.time() - t0, 3)
    result = {
        "three_way": three_way_split,
        "cv_summary": cv_summary,
        "elapsed_sec": elapsed
    }
    _EVAL_CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 2. CLASSIFICATION METRICS (ROC, PR, CONFUSION MATRIX)
# ─────────────────────────────────────────────────────────

def evaluate_classification(model_type="rf"):
    """
    Computes comprehensive classification metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
    ROC curve points, Precision-Recall curve points, and Confusion Matrix.
    """
    if model_type not in ["rf", "gb", "logreg"]:
        model_type = "rf"

    cache_key = f"classification_{model_type}"
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    t0 = time.time()
    data = _get_dataset()
    X = data["X"]
    y = data["y_cls"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=42)
        model_name = "Random Forest Classifier"
    elif model_type == "gb":
        model = GradientBoostingClassifier(n_estimators=70, max_depth=4, random_state=42)
        model_name = "Gradient Boosting Classifier"
    else:
        model = LogisticRegression(max_iter=500, random_state=42)
        model_name = "Logistic Regression (Regularized L2)"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Scalar Metrics
    acc = round(float(accuracy_score(y_test, y_pred)), 4)
    prec = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_test, y_pred)), 4)
    f1 = round(float(f1_score(y_test, y_pred)), 4)
    roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
    pr_auc = round(float(average_precision_score(y_test, y_prob)), 4)

    # ROC Curve
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    # Downsample curve points if too dense for fast browser charting
    step_roc = max(1, len(fpr) // 100)
    roc_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in range(0, len(fpr), step_roc)]
    roc_points.append({"fpr": 1.0, "tpr": 1.0})

    # PR Curve
    precision_curve, recall_curve, pr_thresh = precision_recall_curve(y_test, y_prob)
    step_pr = max(1, len(precision_curve) // 100)
    pr_points = [{"recall": round(float(recall_curve[i]), 4), "precision": round(float(precision_curve[i]), 4)} for i in range(0, len(precision_curve), step_pr)]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0

    cm_data = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": int(len(y_test)),
        "specificity": specificity,
        "raw_matrix": [[tn, fp], [fn, tp]]
    }

    elapsed = round(time.time() - t0, 3)
    result = {
        "model_type": model_type,
        "model_name": model_name,
        "n_test": len(y_test),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "specificity": specificity,
        "roc_points": roc_points,
        "pr_points": pr_points,
        "baseline_pr": round(float(np.mean(y_test)), 4),
        "confusion_matrix": cm_data,
        "elapsed_sec": elapsed
    }
    _EVAL_CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 3. PROBABILITY CALIBRATION & RELIABILITY DIAGRAM
# ─────────────────────────────────────────────────────────

def evaluate_calibration():
    """
    Computes calibration curve (reliability diagram) and Brier score loss
    comparing:
    1. Uncalibrated Model (Random Forest)
    2. Calibrated Model via Platt Scaling / Isotonic (CalibratedClassifierCV)
    3. Perfectly Calibrated Reference Line
    """
    cache_key = "calibration_full"
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    t0 = time.time()
    data = _get_dataset()
    X = data["X"]
    y = data["y_cls"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    # 1. Base Uncalibrated Model
    base_model = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42)
    base_model.fit(X_train, y_train)
    prob_uncal = base_model.predict_proba(X_test)[:, 1]

    # 2. Calibrated Model
    cal_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    cal_model.fit(X_train, y_train)
    prob_cal = cal_model.predict_proba(X_test)[:, 1]

    # Calibration curves (10 bins)
    frac_pos_uncal, mean_prob_uncal = calibration_curve(y_test, prob_uncal, n_bins=10)
    frac_pos_cal, mean_prob_cal = calibration_curve(y_test, prob_cal, n_bins=10)

    # Brier Scores (lower is better; 0 = perfect probability calibration)
    brier_uncal = round(float(brier_score_loss(y_test, prob_uncal)), 4)
    brier_cal = round(float(brier_score_loss(y_test, prob_cal)), 4)
    brier_improvement = round(float(((brier_uncal - brier_cal) / brier_uncal) * 100), 2)

    uncal_points = [{"pred_prob": round(float(p), 4), "actual_fraction": round(float(f), 4)} for p, f in zip(mean_prob_uncal, frac_pos_uncal)]
    cal_points = [{"pred_prob": round(float(p), 4), "actual_fraction": round(float(f), 4)} for p, f in zip(mean_prob_cal, frac_pos_cal)]

    # Probability histograms
    hist_uncal, _ = np.histogram(prob_uncal, bins=10, range=(0, 1))
    hist_cal, _ = np.histogram(prob_cal, bins=10, range=(0, 1))

    bins_labels = [f"{i*10}-{(i+1)*10}%" for i in range(10)]
    hist_data = [{"bin": b, "uncal": int(u), "cal": int(c)} for b, u, c in zip(bins_labels, hist_uncal, hist_cal)]

    elapsed = round(time.time() - t0, 3)
    result = {
        "brier_uncalibrated": brier_uncal,
        "brier_calibrated": brier_cal,
        "brier_improvement_pct": brier_improvement,
        "uncal_points": uncal_points,
        "cal_points": cal_points,
        "hist_data": hist_data,
        "elapsed_sec": elapsed
    }
    _EVAL_CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 4. REGRESSION METRICS (SALARY PACKAGE IN LPA)
# ─────────────────────────────────────────────────────────

def evaluate_regression(model_type="rf_reg"):
    """
    Computes regression evaluation for continuous Salary Package prediction:
    RMSE, MAE, R², MAPE, Residuals vs Predicted scatter coordinates.
    """
    cache_key = f"regression_{model_type}"
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    t0 = time.time()
    data = _get_dataset()
    X = data["X"]
    y = data["y_reg"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if model_type == "rf_reg":
        model = RandomForestRegressor(n_estimators=70, max_depth=6, random_state=42)
        model_name = "Random Forest Regressor"
    else:
        model = Ridge(alpha=1.0)
        model_name = "Ridge Linear Regression"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Core Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = round(float(np.sqrt(mse)), 3)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 3)
    r2 = round(float(r2_score(y_test, y_pred)), 4)
    max_err = round(float(np.max(np.abs(y_test - y_pred))), 3)

    # Mean Absolute Percentage Error (on non-zero actuals)
    non_zero = (y_test > 0.5)
    if non_zero.any():
        mape = round(float(np.mean(np.abs((y_test[non_zero] - y_pred[non_zero]) / y_test[non_zero])) * 100), 2)
    else:
        mape = 0.0

    residuals = y_test - y_pred

    # Scatter points for actual vs predicted and residual plot
    step = max(1, len(y_test) // 500)
    scatter_pts = []
    for i in range(0, len(y_test), step):
        scatter_pts.append({
            "actual": round(float(y_test[i]), 2),
            "predicted": round(float(y_pred[i]), 2),
            "residual": round(float(residuals[i]), 2)
        })

    elapsed = round(time.time() - t0, 3)
    result = {
        "model_name": model_name,
        "model_type": model_type,
        "n_test": len(y_test),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "max_error": max_err,
        "mean_actual": round(float(np.mean(y_test)), 2),
        "scatter_points": scatter_pts,
        "elapsed_sec": elapsed
    }
    _EVAL_CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 5. HYPERPARAMETER SEARCH (GRID vs RANDOM vs BAYESIAN)
# ─────────────────────────────────────────────────────────

def evaluate_tuning():
    """
    Runs and benchmarks 3 hyperparameter search paradigms on Random Forest:
    1. Grid Search (GridSearchCV)
    2. Random Search (RandomizedSearchCV)
    3. Bayesian Optimization (Optuna with TPE Sampler)
    Returns comparative table, best parameters, convergence history, and time metrics.
    """
    cache_key = "tuning_benchmark"
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    t_total = time.time()
    data = _get_dataset(sample_size=1800)
    X = data["X"]
    y = data["y_cls"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 1. Grid Search
    t_grid = time.time()
    param_grid = {
        "n_estimators": [30, 60],
        "max_depth": [4, 7],
        "min_samples_split": [2, 5]
    }
    # 2 x 2 x 2 = 8 combinations * 3 folds = 24 fits
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    grid_time = round(time.time() - t_grid, 3)
    grid_best_score = round(float(grid_search.best_score_), 4)
    grid_test_acc = round(float(accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))), 4)

    # 2. Random Search
    t_rand = time.time()
    param_dist = {
        "n_estimators": [20, 35, 50, 70, 90],
        "max_depth": [3, 5, 7, 9],
        "min_samples_split": [2, 4, 6, 8]
    }
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_dist,
        n_iter=8,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    rand_time = round(time.time() - t_rand, 3)
    rand_best_score = round(float(random_search.best_score_), 4)
    rand_test_acc = round(float(accuracy_score(y_test, random_search.best_estimator_.predict(X_test))), 4)

    # 3. Bayesian Optimization (Optuna with TPE Sampler)
    t_bayes = time.time()
    trial_history = []
    best_so_far = 0.0

    def objective(trial):
        nonlocal best_so_far
        n_estimators = trial.suggest_int("n_estimators", 20, 100, step=10)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8)

        m = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        # 3-fold cross-validation score
        cv_scores = cross_validate(m, X_train, y_train, cv=3, scoring="accuracy")["test_score"]
        mean_s = float(np.mean(cv_scores))

        if mean_s > best_so_far:
            best_so_far = mean_s

        trial_history.append({
            "trial": trial.number + 1,
            "score": round(mean_s, 4),
            "best_so_far": round(best_so_far, 4),
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split
            }
        })
        return mean_s

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=12)
    bayes_time = round(time.time() - t_bayes, 3)
    bayes_best_score = round(float(study.best_value), 4)

    # Train best model on test set
    best_bayes_model = RandomForestClassifier(**study.best_params, random_state=42)
    best_bayes_model.fit(X_train, y_train)
    bayes_test_acc = round(float(accuracy_score(y_test, best_bayes_model.predict(X_test))), 4)

    # Comparative Summary Table
    comparison_table = [
        {
            "strategy": "Grid Search",
            "technique": "Exhaustive Parameter Permutation",
            "evaluations": 8,
            "best_cv_score": grid_best_score,
            "test_accuracy": grid_test_acc,
            "time_sec": grid_time,
            "best_params": str(grid_search.best_params_),
            "efficiency": "Brute Force (High compute cost)"
        },
        {
            "strategy": "Random Search",
            "technique": "Uniform Stochastic Sampling",
            "evaluations": 8,
            "best_cv_score": rand_best_score,
            "test_accuracy": rand_test_acc,
            "time_sec": rand_time,
            "best_params": str(random_search.best_params_),
            "efficiency": "Higher probability of finding optima in wide search space"
        },
        {
            "strategy": "Bayesian Optimization",
            "technique": "Tree-structured Parzen Estimator (TPE)",
            "evaluations": 12,
            "best_cv_score": bayes_best_score,
            "test_accuracy": bayes_test_acc,
            "time_sec": bayes_time,
            "best_params": str(study.best_params),
            "efficiency": "Highest efficiency: Models objective function probabilistically"
        }
    ]

    total_time = round(time.time() - t_total, 3)
    result = {
        "comparison_table": comparison_table,
        "trial_history": trial_history,
        "grid_best_params": grid_search.best_params_,
        "rand_best_params": random_search.best_params_,
        "bayes_best_params": study.best_params,
        "total_elapsed_sec": total_time
    }
    _EVAL_CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────
# 6. COMPREHENSIVE SCORECARD SUMMARY
# ─────────────────────────────────────────────────────────

def evaluate_summary():
    """
    Returns an executive summary scorecard across all evaluation disciplines.
    """
    splits_res = evaluate_splits(cv_folds=5)
    cls_res = evaluate_classification(model_type="rf")
    cal_res = evaluate_calibration()
    reg_res = evaluate_regression(model_type="rf_reg")
    tuning_res = evaluate_tuning()

    return {
        "splits": splits_res["three_way"],
        "cv": splits_res["cv_summary"],
        "classification": cls_res,
        "calibration": cal_res,
        "regression": reg_res,
        "tuning": tuning_res
    }
