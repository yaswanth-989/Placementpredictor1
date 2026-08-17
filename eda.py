import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')         # No display needed — Flask runs headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from load_data import load_data

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "Static", "plots")


def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Shared plot style (light theme)
# ─────────────────────────────────────────
PLOT_STYLE = {
    "facecolor":    "#ffffff",
    "grid_color":   "#e2e8f0",
    "spine_color":  "#e2e8f0",
    "label_color":  "#475569",
    "title_color":  "#0f172a",
    "bar_color":    "#2563eb",
    "bar_edge":     "#1d4ed8",
    "tick_color":   "#64748b",
}


def _apply_light_axes(ax, fig):
    """Apply the light-mode style to a matplotlib axes."""
    fig.patch.set_facecolor(PLOT_STYLE["facecolor"])
    ax.set_facecolor(PLOT_STYLE["facecolor"])
    ax.tick_params(colors=PLOT_STYLE["tick_color"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(PLOT_STYLE["spine_color"])
    ax.yaxis.grid(True, color=PLOT_STYLE["grid_color"],
                  linestyle="--", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────
# Statistical Summary
# ─────────────────────────────────────────

def get_statistical_summary():
    """
    Returns:
      numeric     – describe(numeric_only=True) restructured for template
      categorical – count / unique / top / freq per categorical column
    """
    df = load_data()

    # Numeric
    numeric_desc = df.describe(numeric_only=True)
    numeric = {
        "columns": list(numeric_desc.columns),
        "stats":   list(numeric_desc.index),
        "data":    {}
    }
    for col in numeric_desc.columns:
        numeric["data"][col] = {}
        for stat in numeric_desc.index:
            val = numeric_desc.loc[stat, col]
            if pd.isna(val):
                numeric["data"][col][stat] = "-"
            else:
                numeric["data"][col][stat] = round(float(val), 4)

    # Categorical
    cat_cols  = df.select_dtypes(exclude="number").columns.tolist()
    cat_stats = []
    for col in cat_cols:
        vc = df[col].value_counts()
        cat_stats.append({
            "column": col,
            "count":  int(df[col].count()),
            "unique": int(df[col].nunique()),
            "top":    str(vc.index[0]) if len(vc) > 0 else "-",
            "freq":   int(vc.iloc[0])  if len(vc) > 0 else 0
        })

    return {"numeric": numeric, "categorical": cat_stats}


# ─────────────────────────────────────────
# Unique Value Counts
# ─────────────────────────────────────────

def get_unique_counts():
    """Returns {column, unique_count, dtype} for every column."""
    df = load_data()
    return [
        {
            "column":       col,
            "unique_count": int(df[col].nunique()),
            "dtype":        str(df[col].dtype)
        }
        for col in df.columns
    ]


# ─────────────────────────────────────────
# Value Counts for a single column
# ─────────────────────────────────────────

def get_value_counts(col, top_n=20):
    """Top N value counts for one column."""
    df = load_data()
    if col not in df.columns:
        return []
    vc = df[col].value_counts().head(top_n)
    return [{"value": str(k), "count": int(v)} for k, v in vc.items()]


# ─────────────────────────────────────────
# All column names  (for dropdown)
# ─────────────────────────────────────────

def get_columns():
    return list(load_data().columns)


# ─────────────────────────────────────────
# Correlation Heatmap
# ─────────────────────────────────────────

def generate_correlation_heatmap(force=False):
    """
    Saves a light-themed correlation heatmap.
    Returns the static-relative path, e.g. 'plots/corr_heatmap.png'.
    """
    _ensure_plots_dir()
    out = os.path.join(PLOTS_DIR, "corr_heatmap.png")
    if os.path.exists(out) and not force:
        return "plots/corr_heatmap.png"

    df   = load_data()
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    sns.heatmap(
        corr, annot=True, cmap="Blues",
        fmt=".2f", linewidths=0.6, linecolor="#e2e8f0",
        vmin=-1, vmax=1,
        annot_kws={"size": 8.5, "color": "#1e293b"},
        ax=ax, cbar_kws={"shrink": 0.75}
    )
    ax.set_title("Correlation Heatmap", color=PLOT_STYLE["title_color"],
                 fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors=PLOT_STYLE["tick_color"], labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    return "plots/corr_heatmap.png"


# ─────────────────────────────────────────
# Missing Values Bar Chart
# ─────────────────────────────────────────

def generate_missing_bar_chart(force=False):
    """
    Saves a bar chart of missing-value percentages.
    Returns relative path, or None if no missing values.
    """
    _ensure_plots_dir()
    out = os.path.join(PLOTS_DIR, "missing_bar.png")
    if os.path.exists(out) and not force:
        return "plots/missing_bar.png"

    df          = load_data()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if missing_pct.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_light_axes(ax, fig)

    bars = ax.bar(
        missing_pct.index, missing_pct.values,
        color=PLOT_STYLE["bar_color"],
        edgecolor=PLOT_STYLE["bar_edge"],
        linewidth=0.7, width=0.6
    )

    # Value labels on bars
    for bar, val in zip(bars, missing_pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            "{:.2f}%".format(val),
            ha="center", va="bottom",
            fontsize=8, color=PLOT_STYLE["label_color"]
        )

    ax.set_title("Missing Value % per Column",
                 color=PLOT_STYLE["title_color"], fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Column",             color=PLOT_STYLE["label_color"], fontsize=10)
    ax.set_ylabel("Missing Percentage", color=PLOT_STYLE["label_color"], fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    return "plots/missing_bar.png"


# ─────────────────────────────────────────
# Histograms  (one per numeric column)
# ─────────────────────────────────────────

def generate_histograms(force=False, max_cols=10):
    """
    Generates histogram PNGs for up to max_cols numeric columns.
    Returns list of {column, path} dicts.
    """
    _ensure_plots_dir()
    df           = load_data()
    numeric_cols = list(df.select_dtypes(include="number").columns)[:max_cols]
    paths        = []

    for col in numeric_cols:
        fname = "hist_{}.png".format(
            col.replace(" ", "_").replace("/", "_")
        )
        out = os.path.join(PLOTS_DIR, fname)

        if not os.path.exists(out) or force:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _apply_light_axes(ax, fig)

            ax.hist(df[col].dropna(), bins=30,
                    color=PLOT_STYLE["bar_color"],
                    edgecolor="#ffffff", linewidth=0.5)
            ax.set_title("Distribution of {}".format(col),
                         color=PLOT_STYLE["title_color"],
                         fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel(col,         color=PLOT_STYLE["label_color"], fontsize=10)
            ax.set_ylabel("Frequency", color=PLOT_STYLE["label_color"], fontsize=10)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: "{:,.0f}".format(x)
            ))

            plt.tight_layout()
            plt.savefig(out, dpi=110, bbox_inches="tight", facecolor="#ffffff")
            plt.close()

        paths.append({"column": col, "path": "plots/{}".format(fname)})

    return paths
