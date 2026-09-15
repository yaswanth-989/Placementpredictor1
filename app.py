from flask import Flask, render_template, request, redirect, url_for

from load_data import (
    get_data_summary,
    get_duplicate_count,
    get_eda_summary,
    get_columns_metadata,
    get_plot_data
)
import preprocess
import eda as eda_module
import feature_scaling as fs_module
import encoding as enc_module
import linear_regression as lr_module
import decision_tree_module as dt_module
import ensemble_models as ens_module
import boosting_models as boost_module
import predict as predict_module

app = Flask(__name__)


# ─────────────────────────────────────────
# Home / Dashboard
# ─────────────────────────────────────────

@app.route("/")
def index():
    # Pass live dataset stats to dashboard
    total_rows = 0
    total_cols = 0
    try:
        summary    = get_data_summary()
        total_rows = summary["n_rows"]
        total_cols = summary["n_cols"]
    except Exception:
        pass
    return render_template(
        "index.html",
        active="home",
        total_rows=total_rows,
        total_cols=total_cols
    )


# ─────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────

@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    duplicate_count = None
    try:
        summary         = get_data_summary()
        duplicate_count = get_duplicate_count()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = "Unexpected error: {}".format(e)

    return render_template(
        "data_loading.html",
        active          = "data-loading",
        summary         = summary,
        duplicate_count = duplicate_count,
        error           = error
    )


# ─────────────────────────────────────────
# EDA  (all data computed server-side)
# ─────────────────────────────────────────

@app.route("/eda")
def eda():
    error             = None
    eda_summary       = None
    stat_summary      = None
    unique_counts     = []
    value_counts_data = []
    columns_meta      = None
    plot_data         = None
    columns           = []
    corr_heatmap      = None
    missing_bar       = None
    histograms        = []

    # User selections from query string
    viz_col      = request.args.get("viz_col",      "CGPA")
    chart_type   = request.args.get("chart_type",   "histogram")
    selected_vc_col = request.args.get("vc_col",    None)
    force_refresh = (request.args.get("refresh") == "1")

    # Column list & metadata
    try:
        columns      = eda_module.get_columns()
        columns_meta = get_columns_metadata()
    except Exception as e:
        error = str(e)

    # EDA overview summary
    try:
        eda_summary = get_eda_summary()
    except Exception as e:
        error = str(e)

    # Statistical summary + unique counts
    try:
        stat_summary  = eda_module.get_statistical_summary()
        unique_counts = eda_module.get_unique_counts()
    except Exception as e:
        error = str(e)

    # Value counts dropdown
    try:
        if not selected_vc_col:
            selected_vc_col = columns[0] if columns else None
        if selected_vc_col and selected_vc_col in columns:
            value_counts_data = eda_module.get_value_counts(selected_vc_col)
    except Exception as e:
        error = str(e)

    # Interactive Plotly data — computed server-side, embedded as JSON
    try:
        all_cols = columns_meta.get("all", []) if columns_meta else []
        if viz_col not in all_cols and all_cols:
            num_cols = columns_meta.get("numerical", [])
            viz_col  = num_cols[0] if num_cols else all_cols[0]
        plot_data = get_plot_data(column=viz_col, chart_type=chart_type)
    except Exception as e:
        error = str(e)

    # Static cached plots (matplotlib PNGs)
    try:
        corr_heatmap = eda_module.generate_correlation_heatmap(force=force_refresh)
    except Exception:
        pass

    try:
        missing_bar = eda_module.generate_missing_bar_chart(force=force_refresh)
    except Exception:
        pass

    try:
        histograms = eda_module.generate_histograms(force=force_refresh)
    except Exception:
        pass

    return render_template(
        "eda.html",
        active            = "eda",
        error             = error,
        eda_summary       = eda_summary,
        stat_summary      = stat_summary,
        unique_counts     = unique_counts,
        value_counts_data = value_counts_data,
        selected_vc_col   = selected_vc_col,
        columns           = columns,
        columns_meta      = columns_meta,
        plot_data         = plot_data,
        viz_col           = viz_col,
        chart_type        = chart_type,
        corr_heatmap      = corr_heatmap,
        missing_bar       = missing_bar,
        histograms        = histograms
    )


# ─────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────

@app.route("/preprocessing")
def preprocessing():
    error           = None
    action          = request.args.get("action", "missing-summary")
    result          = None
    missing_summary = None

    try:
        missing_summary = preprocess.get_missing_summary()

        if action == "drop-missing":
            result = preprocess.drop_missing()
        elif action == "mean-imputation":
            result = preprocess.mean_imputation()
        elif action == "median-imputation":
            result = preprocess.median_imputation()
        elif action == "mode-imputation":
            result = preprocess.mode_imputation()

    except Exception as e:
        error = "Error: {}".format(e)

    return render_template(
        "preprocessing.html",
        active          = "preprocessing",
        action          = action,
        missing_summary = missing_summary,
        result          = result,
        error           = error
    )


# ─────────────────────────────────────────
# Feature Scaling
# ─────────────────────────────────────────

@app.route("/feature-scaling")
def feature_scaling():
    error      = None
    action     = request.args.get("action", "original")
    result     = None
    orig_stats = None
    comparison = None

    try:
        orig_stats = fs_module.get_original_stats()
        comparison = fs_module.get_scaling_comparison()

        if action == "minmax":
            result = fs_module.minmax_scaling()
        elif action == "standard":
            result = fs_module.standard_scaling()
        elif action == "robust":
            result = fs_module.robust_scaling()

    except Exception as e:
        error = "Error: {}".format(e)

    return render_template(
        "feature_scaling.html",
        active     = "feature-scaling",
        action     = action,
        result     = result,
        orig_stats = orig_stats,
        comparison = comparison,
        error      = error
    )


# ─────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────

@app.route("/encoding")
def encoding():
    error       = None
    action      = request.args.get("action", "info")
    result      = None
    cat_info    = None
    enc_summary = None

    try:
        cat_info    = enc_module.get_categorical_info()
        enc_summary = enc_module.get_encoding_summary()

        if action == "label":
            result = enc_module.label_encoding()
        elif action == "onehot":
            result = enc_module.onehot_encoding()
        elif action == "ordinal":
            result = enc_module.ordinal_encoding()

    except Exception as e:
        error = "Error: {}".format(e)

    return render_template(
        "encoding.html",
        active      = "encoding",
        action      = action,
        result      = result,
        cat_info    = cat_info,
        enc_summary = enc_summary,
        error       = error
    )


# ─────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────

@app.route("/linear-regression")
def linear_regression():
    error    = None
    action   = request.args.get("action", "overview")
    result   = None
    overview = None

    try:
        overview = lr_module.get_lr_overview()

        if action == "simple":
            result = lr_module.simple_linear_regression()
        elif action == "multiple":
            result = lr_module.multiple_linear_regression()
        elif action == "leakage":
            result = lr_module.data_leakage_demo()

    except Exception as e:
        error = "Error: {}".format(e)

    return render_template(
        "linear_regression.html",
        active   = "linear-regression",
        action   = action,
        result   = result,
        overview = overview,
        error    = error
    )


# ─────────────────────────────────────────
# Decision Tree
# ─────────────────────────────────────────

@app.route("/decision-tree")
def decision_tree():
    error       = None
    action      = request.args.get("action", "concepts")
    result      = None

    try:
        if action == "concepts":
            result = dt_module.entropy_gini_demo()
        elif action == "train":
            result = dt_module.train_decision_tree()
        elif action == "compare":
            result = dt_module.model_comparison()

    except Exception as e:
        error = "Error: {}".format(e)

    return render_template(
        "decision_tree.html",
        active  = "decision-tree",
        action  = action,
        result  = result,
        error   = error
    )


# ─────────────────────────────────────────
# Ensemble Models
# ─────────────────────────────────────────

@app.route("/ensemble-models")
def ensemble_models():
    error  = None
    action = request.args.get("action", "rf")
    result = None
    try:
        if action == "rf":
            result = ens_module.random_forest()
        elif action == "adaboost":
            result = ens_module.adaboost()
        elif action == "compare":
            result = ens_module.ensemble_comparison()
    except Exception as e:
        error = "Error: {}".format(e)
    return render_template(
        "ensemble_models.html",
        active = "ensemble-models",
        action = action,
        result = result,
        error  = error
    )


# ─────────────────────────────────────────
# Boosting Models
# ─────────────────────────────────────────

@app.route("/boosting-models")
def boosting_models():
    error  = None
    action = request.args.get("action", "gb")
    result = None
    try:
        if action == "gb":
            result = boost_module.gradient_boosting()
        elif action == "xgboost":
            result = boost_module.xgboost_model()
        elif action == "lightgbm":
            result = boost_module.lightgbm_model()
        elif action == "compare":
            result = boost_module.boosting_comparison()
    except Exception as e:
        error = "Error: {}".format(e)
    return render_template(
        "boosting_models.html",
        active = "boosting-models",
        action = action,
        result = result,
        error  = error
    )


# ─────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    error     = None
    result    = None
    form_data = None
    meta      = predict_module.get_form_meta()

    if request.method == "POST":
        form_data = dict(request.form)
        model_name = form_data.get("model_name", "random_forest")
        # Flatten single-value lists from form multidict
        form_data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v
                     for k, v in form_data.items()}
        try:
            result = predict_module.predict_placement(form_data, model_name)
        except Exception as e:
            error = "Prediction error: {}".format(e)

    return render_template(
        "prediction.html",
        active    = "prediction",
        meta      = meta,
        form_data = form_data,
        result    = result,
        error     = error
    )


# ─────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)