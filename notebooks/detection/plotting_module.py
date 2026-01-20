# plotting_module.py
# Improved plotting module for evaluating segmentation experiments.
# Keeps dependencies minimal (matplotlib/numpy/pandas) and preserves the original API.

import os
import json
import math
import textwrap
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Global style
# ============================================================

def _apply_style() -> None:
    """Set a consistent, thesis-friendly matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

_apply_style()


# ============================================================
# Utility helpers
# ============================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _wrap_label(s: str, width: int = 18) -> str:
    # Avoid breaking long experiment names into unreadable single words
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False))


def _save_fig(fig: plt.Figure, out_path: str) -> None:
    """Save both PNG and PDF with tight layout."""
    _ensure_dir(os.path.dirname(out_path) or ".")
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _fd_bins(x: np.ndarray, fallback: int = 30) -> np.ndarray:
    """Freedman–Diaconis binning; stable fallback if data is degenerate."""
    if x.size < 2:
        return np.array([0.0, 1.0])
    try:
        bins = np.histogram_bin_edges(x, bins="fd")
        if len(bins) < 5:  # too few bins looks bad
            bins = np.histogram_bin_edges(x, bins=fallback)
        return bins
    except Exception:
        return np.histogram_bin_edges(x, bins=fallback)


def _bounded_01(col: str) -> bool:
    return col.lower() in {"dice", "iou", "precision", "recall", "f1"}


def _lower_is_better(metric: str) -> bool:
    m = metric.lower()
    return ("error" in m) or (m in {"avg_infer_time"})


def _format_value(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "NA"
    # Nice formatting for typical ranges
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


# ============================================================
# Load experiment results
# ============================================================

def load_experiment(exp_dir: str) -> Dict:
    """
    Loads:
      - results.json (global metrics)
      - test_semantic_metrics.csv (per-image metrics)

    Returns:
      {
        "name": <exp_name>,
        "json": <dict>,
        "df": <pandas.DataFrame>,
        "plots_dir": <path>
      }
    """
    json_path = os.path.join(exp_dir, "results.json")
    csv_path = os.path.join(exp_dir, "test_semantic_metrics.csv")
    plots_dir = os.path.join(exp_dir, "plots")
    _ensure_dir(plots_dir)

    with open(json_path, "r") as f:
        res_json = json.load(f)

    df = pd.read_csv(csv_path)
    name = os.path.basename(os.path.normpath(exp_dir))

    return {"name": name, "json": res_json, "df": df, "plots_dir": plots_dir}


# ============================================================
# Per-experiment plotting
# ============================================================

def plot_distribution(
    df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    out_path: str,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    show_mean_median: bool = True
) -> None:
    s = _to_numeric_series(df, column)
    if s.empty:
        return

    x = s.to_numpy()
    bins = _fd_bins(x, fallback=30)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist(x, bins=bins, alpha=0.9, edgecolor="white", linewidth=0.6)

    if _bounded_01(column):
        ax.set_xlim(0.0, 1.0)
    elif xlim is not None:
        ax.set_xlim(*xlim)

    if show_mean_median:
        mu = float(np.mean(x))
        md = float(np.median(x))
        ax.axvline(mu, linestyle="-", linewidth=1.2, alpha=0.9, label=f"mean = {_format_value(mu)}")
        ax.axvline(md, linestyle="--", linewidth=1.2, alpha=0.9, label=f"median = {_format_value(md)}")
        ax.legend(loc="best", frameon=False)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.text(0.98, 0.96, f"n = {len(x)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, alpha=0.85)

    _save_fig(fig, out_path)


def plot_bubble_area_distribution(df: pd.DataFrame, out_path: str) -> None:
    # Prefer GT_MeanArea if present, else MeanArea
    col = "MeanArea" if "MeanArea" in df.columns else None
    if col is None:
        return

    s = _to_numeric_series(df, col)
    s = s[s > 0]
    if s.empty:
        return

    x = s.to_numpy()
    bins = _fd_bins(x, fallback=40)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist(x, bins=bins, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.set_title("Bubble Area Distribution (Mean Area per Image)")
    ax.set_xlabel("Area (pixels)")
    ax.set_ylabel("Count")

    # If the distribution is heavy-tailed, a log-x plot is often clearer; save a second figure.
    _save_fig(fig, out_path)

    # Optional log-scale version (same data, different readability)
    if np.max(x) / max(np.min(x), 1.0) > 50:
        fig2, ax2 = plt.subplots(figsize=(6.6, 4.2))
        ax2.hist(x, bins=bins, alpha=0.9, edgecolor="white", linewidth=0.6)
        ax2.set_xscale("log")
        ax2.set_title("Bubble Area Distribution (log scale)")
        ax2.set_xlabel("Area (pixels, log scale)")
        ax2.set_ylabel("Count")
        _save_fig(fig2, out_path + "_log")


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str
) -> None:
    df_valid = df[[x_col, y_col]].copy()
    df_valid[x_col] = pd.to_numeric(df_valid[x_col], errors="coerce")
    df_valid[y_col] = pd.to_numeric(df_valid[y_col], errors="coerce")
    df_valid = df_valid.dropna()
    if df_valid.empty:
        return

    x = df_valid[x_col].to_numpy()
    y = df_valid[y_col].to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(x, y, alpha=0.75, edgecolor="none")

    # y=x reference
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, alpha=0.7)

    # correlation (Pearson)
    if len(x) >= 2:
        r = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.02, 0.98, f"Pearson r = {_format_value(r)}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    _save_fig(fig, out_path)


def plot_timing(json_data: Dict, out_path: str) -> None:
    timing = json_data.get("timing", {})
    fps = float(timing.get("fps", float("nan")))
    avg_t_ms = float(timing.get("avg_infer_time", float("nan"))) * 1000.0

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    labels = ["FPS", "Avg infer (ms)"]
    vals = [fps, avg_t_ms]

    ax.bar(labels, vals, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.set_title("Inference speed summary")
    ax.set_ylabel("Value")

    for i, v in enumerate(vals):
        ax.text(i, v, _format_value(v), ha="center", va="bottom", fontsize=9, alpha=0.85)

    _save_fig(fig, out_path)


def generate_single_experiment_plots(exp_dir: str) -> None:
    """Generate all standard plots for one experiment."""
    data = load_experiment(exp_dir)
    df = data["df"]
    plots = data["plots_dir"]

    print(f"Generating plots for {data['name']} ...")

    # Semantic distributions
    plot_distribution(df, "Dice", "Dice distribution", "Dice", os.path.join(plots, "dice_dist"))
    plot_distribution(df, "IoU", "IoU distribution", "IoU", os.path.join(plots, "iou_dist"))
    plot_distribution(df, "Precision", "Precision distribution", "Precision", os.path.join(plots, "precision_dist"))
    plot_distribution(df, "Recall", "Recall distribution", "Recall", os.path.join(plots, "recall_dist"))
    plot_distribution(df, "F1", "F1 distribution", "F1", os.path.join(plots, "f1_dist"))

    # Instance-level distributions
    if "NumBubbles" in df.columns:
        plot_distribution(df, "NumBubbles", "Predicted bubble count per image", "Pred NumBubbles",
                          os.path.join(plots, "bubble_count_dist"), show_mean_median=True)
    plot_bubble_area_distribution(df, os.path.join(plots, "bubble_area_dist"))

    # Optional GT-aware plots
    if "GT_NumBubbles" in df.columns:
        plot_distribution(df, "GT_NumBubbles", "GT bubble count per image", "GT NumBubbles",
                          os.path.join(plots, "gt_bubble_count_dist"), show_mean_median=True)

    if "CountError" in df.columns:
        # Center around 0 for readability
        ce = _to_numeric_series(df, "CountError")
        if not ce.empty:
            lim = float(max(abs(ce.min()), abs(ce.max())))
            plot_distribution(df, "CountError", "Bubble count error (Pred - GT)", "CountError",
                              os.path.join(plots, "count_error_dist"),
                              xlim=(-lim, lim), show_mean_median=True)
            # add a 0-reference line by re-plotting quickly (keeps API simple)
            # (If you want it baked in, we can add a specialized function.)
    if "AbsCountError" in df.columns:
        plot_distribution(df, "AbsCountError", "Absolute bubble count error", "AbsCountError",
                          os.path.join(plots, "abs_count_error_dist"), show_mean_median=True)

    if "GT_MeanArea" in df.columns:
        df_gt_area = df[pd.to_numeric(df["GT_MeanArea"], errors="coerce") > 0]
        if not df_gt_area.empty:
            plot_distribution(df_gt_area, "GT_MeanArea", "GT mean bubble area per image", "GT_MeanArea (pixels)",
                              os.path.join(plots, "gt_bubble_area_dist"), show_mean_median=True)

    if "GT_NumBubbles" in df.columns and "NumBubbles" in df.columns:
        plot_scatter(df, "GT_NumBubbles", "NumBubbles",
                     "Pred vs GT bubble count", "GT NumBubbles", "Pred NumBubbles",
                     os.path.join(plots, "pred_vs_gt_bubble_count"))

    # Timing summary
    plot_timing(data["json"], os.path.join(plots, "timing"))

    print(f"Plots saved to {plots}")


# ============================================================
# Multi-experiment comparison
# ============================================================

def _get_metric_from_results(j: Dict, metric: str) -> float:
    if "semantic_metrics" in j and metric in j["semantic_metrics"]:
        return float(j["semantic_metrics"][metric])
    if "instance_metrics" in j and metric in j["instance_metrics"]:
        return float(j["instance_metrics"][metric])
    if "timing" in j and metric in j["timing"]:
        return float(j["timing"][metric])
    raise ValueError(f"Metric '{metric}' not found in results.json")


def plot_metric_comparison(
    experiments: List[str],
    metric: str,
    title: str,
    ylabel: str,
    out_path: str
) -> None:
    """
    Thesis-friendly comparison plot across experiments:
    - horizontal bars
    - sorted (best on top)
    - value labels
    """
    rows = []
    for exp in experiments:
        data = load_experiment(exp)
        name = data["name"]
        val = _get_metric_from_results(data["json"], metric)
        rows.append((name, val))

    # Sort
    lower_better = _lower_is_better(metric)
    rows.sort(key=lambda t: t[1], reverse=(not lower_better))

    names = [_wrap_label(r[0], width=24) for r in rows]
    values = [r[1] for r in rows]

    fig_h = max(3.6, 0.42 * len(names))
    fig, ax = plt.subplots(figsize=(8.2, fig_h))

    y = np.arange(len(names))
    ax.barh(y, values, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # best at top
    ax.set_title(title)
    ax.set_xlabel(ylabel)

    # Add value labels
    for yi, v in zip(y, values):
        ax.text(v, yi, f"  {_format_value(v)}", va="center", ha="left", fontsize=9, alpha=0.9)

    _save_fig(fig, out_path)


def generate_experiment_comparison(experiments: List[str], out_dir: str = "comparisons") -> None:
    """Takes a list of experiment directories and generates comparison plots."""
    _ensure_dir(out_dir)

    print("Comparing experiments:")
    for e in experiments:
        print(" -", e)

    # Basic semantic metrics comparisons
    plot_metric_comparison(experiments, "dice", "Mean Dice across experiments", "Dice",
                           os.path.join(out_dir, "compare_dice"))
    plot_metric_comparison(experiments, "iou", "Mean IoU across experiments", "IoU",
                           os.path.join(out_dir, "compare_iou"))
    plot_metric_comparison(experiments, "precision", "Mean Precision across experiments", "Precision",
                           os.path.join(out_dir, "compare_precision"))
    plot_metric_comparison(experiments, "recall", "Mean Recall across experiments", "Recall",
                           os.path.join(out_dir, "compare_recall"))
    plot_metric_comparison(experiments, "f1", "Mean F1 across experiments", "F1",
                           os.path.join(out_dir, "compare_f1"))

    # Instance-level metrics
    plot_metric_comparison(experiments, "avg_bubbles", "Average predicted bubble count", "Avg bubbles",
                           os.path.join(out_dir, "compare_bubble_count"))
    plot_metric_comparison(experiments, "abs_count_error_mean", "Mean absolute bubble count error", "Abs count error",
                           os.path.join(out_dir, "compare_abs_count_error"))
    plot_metric_comparison(experiments, "count_error_mean", "Mean bubble count error (Pred - GT)", "Count error",
                           os.path.join(out_dir, "compare_count_error"))

    # Inference speed
    plot_metric_comparison(experiments, "fps", "Inference FPS comparison", "FPS",
                           os.path.join(out_dir, "compare_fps"))

    # Optional: also save a summary table for the thesis
    summary_rows = []
    for exp in experiments:
        data = load_experiment(exp)
        j = data["json"]
        summary_rows.append({
            "experiment": data["name"],
            "dice": _get_metric_from_results(j, "dice"),
            "iou": _get_metric_from_results(j, "iou"),
            "abs_count_error_mean": _get_metric_from_results(j, "abs_count_error_mean"),
            "count_error_mean": _get_metric_from_results(j, "count_error_mean"),
            "fps": _get_metric_from_results(j, "fps"),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)


# ============================================================
# Convenience function
# ============================================================

def run_all_plots_for_experiment(exp_dir: str) -> None:
    """One-call function to process a single experiment directory."""
    generate_single_experiment_plots(exp_dir)
    print("Finished plotting for:", exp_dir)
