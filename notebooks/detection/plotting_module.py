# plotting_module.py
# A reusable plotting module for evaluating segmentation experiments.
# Generates thesis-ready distribution plots and cross-experiment comparisons.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


# ============================================================
# Utility helpers
# ============================================================
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_fig(fig, out_path):
    png_path = out_path + ".png"
    pdf_path = out_path + ".pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Load experiment results
# ============================================================
def load_experiment(exp_dir):
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

    return {
        "name": name,
        "json": res_json,
        "df": df,
        "plots_dir": plots_dir
    }


# ============================================================
# Per-experiment plotting
# ============================================================

def plot_distribution(df, column, title, xlabel, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[column], bins=40, density=False, alpha=0.8, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_path)


def plot_bubble_area_distribution(df, out_path):
    df_valid = df[df["MeanArea"] > 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df_valid["MeanArea"], bins=40, alpha=0.8, color="darkorange")
    ax.set_title("Bubble Area Distribution (Mean Area per Image)")
    ax.set_xlabel("Area (pixels)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_path)


def plot_timing(json_data, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    fps = json_data["timing"]["fps"]
    avg_t = json_data["timing"]["avg_infer_time"] * 1000  # ms

    ax.bar(["FPS", "Avg Time (ms)"], [fps, avg_t], color=["green", "red"])
    ax.set_title("Inference Speed")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_path)


def generate_single_experiment_plots(exp_dir):
    """
    Generate all standard plots for one experiment.
    """
    data = load_experiment(exp_dir)
    df = data["df"]
    plots = data["plots_dir"]

    print(f"Generating plots for {data['name']} ...")

    # Semantic distributions
    plot_distribution(df, "Dice", "Dice Distribution", "Dice", os.path.join(plots, "dice_dist"))
    plot_distribution(df, "IoU", "IoU Distribution", "IoU", os.path.join(plots, "iou_dist"))
    plot_distribution(df, "Precision", "Precision Distribution", "Precision", os.path.join(plots, "precision_dist"))
    plot_distribution(df, "Recall", "Recall Distribution", "Recall", os.path.join(plots, "recall_dist"))
    plot_distribution(df, "F1", "F1 Distribution", "F1", os.path.join(plots, "f1_dist"))

    # Instance-level distributions
    plot_distribution(df, "NumBubbles", "Bubble Count per Image", "Num Bubbles",
                      os.path.join(plots, "bubble_count_dist"))
    plot_bubble_area_distribution(df, os.path.join(plots, "bubble_area_dist"))

    # Timing summary
    plot_timing(data["json"], os.path.join(plots, "timing"))

    print(f"Plots saved to {plots}")


# ============================================================
# Multi-experiment comparison
# ============================================================

def plot_metric_comparison(experiments: List[str], metric: str, title: str, ylabel: str, out_path: str):
    """
    Bar plot comparing a given metric across experiments.
    Metric must be in results.json under semantic_metrics or instance_metrics.
    """
    names = []
    values = []

    for exp in experiments:
        data = load_experiment(exp)
        j = data["json"]

        if metric in j["semantic_metrics"]:
            val = j["semantic_metrics"][metric]
        elif metric in j["instance_metrics"]:
            val = j["instance_metrics"][metric]
        elif metric in j["timing"]:
            val = j["timing"][metric]
        else:
            raise ValueError(f"Metric {metric} not found in {exp}")

        names.append(data["name"])
        values.append(val)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color="royalblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=20)
    _save_fig(fig, out_path)


def generate_experiment_comparison(experiments: List[str], out_dir="comparisons"):
    """
    Takes a list of experiment directories and generates comparison plots.
    """
    _ensure_dir(out_dir)

    print("Comparing experiments:")
    for e in experiments:
        print(" -", e)

    # Basic semantic metrics comparisons
    plot_metric_comparison(
        experiments,
        "dice",
        "Mean Dice Across Experiments",
        "Dice",
        os.path.join(out_dir, "compare_dice"),
    )
    plot_metric_comparison(
        experiments,
        "iou",
        "Mean IoU Across Experiments",
        "IoU",
        os.path.join(out_dir, "compare_iou"),
    )
    plot_metric_comparison(
        experiments,
        "precision",
        "Mean Precision Across Experiments",
        "Precision",
        os.path.join(out_dir, "compare_precision"),
    )
    plot_metric_comparison(
        experiments,
        "recall",
        "Mean Recall Across Experiments",
        "Recall",
        os.path.join(out_dir, "compare_recall"),
    )
    plot_metric_comparison(
        experiments,
        "f1",
        "Mean F1 Across Experiments",
        "F1 Score",
        os.path.join(out_dir, "compare_f1"),
    )

    # Instance-level metrics
    plot_metric_comparison(
        experiments,
        "avg_bubbles",
        "Average Bubble Count",
        "Avg Bubbles",
        os.path.join(out_dir, "compare_bubble_count"),
    )

    # Inference speed
    plot_metric_comparison(
        experiments,
        "fps",
        "Inference FPS Comparison",
        "FPS",
        os.path.join(out_dir, "compare_fps"),
    )


# ============================================================
# Convenience function
# ============================================================

def run_all_plots_for_experiment(exp_dir):
    """
    One-call function to process a single experiment directory.
    """
    generate_single_experiment_plots(exp_dir)
    print("Finished plotting for:", exp_dir)
