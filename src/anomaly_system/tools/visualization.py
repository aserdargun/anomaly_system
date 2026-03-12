"""Visualization functions — optimized for fast LLM vision recognition."""

import matplotlib
matplotlib.use("Agg")  # Must be before pyplot import — headless rendering on macOS

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

from anomaly_system.config import PLOT_DPI, PLOT_FIGSIZE

# High-contrast style optimized for LLM vision at low resolution
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
})


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> str:
    """Plot confusion matrix as a seaborn heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Path to save PNG.

    Returns:
        Path to saved file.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    # Simple table-style — no heatmap gradient, just numbers on solid colors
    # This is faster for LLM vision than a gradient heatmap
    labels = np.array([
        [f"TN={cm[0,0]}", f"FP={cm[0,1]}"],
        [f"FN={cm[1,0]}", f"TP={cm[1,1]}"],
    ])
    colors = np.array([
        ["#C8E6C9", "#FFCDD2"],  # TN=green, FP=red
        ["#FFCDD2", "#C8E6C9"],  # FN=red, TP=green
    ])

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, facecolor=colors[i,j], edgecolor="white", lw=2))
            ax.text(j+0.5, 1.5-i, labels[i,j], ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#212121")

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Anomaly", "Normal"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    return save_path


def plot_score_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    save_path: str,
) -> str:
    """Plot anomaly score distributions for normal vs anomaly classes.

    Args:
        y_true: True labels.
        scores: Anomaly scores from the model.
        save_path: Path to save PNG.

    Returns:
        Path to saved file.
    """
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    # Fewer bins, bold colors, high contrast
    ax.hist(normal_scores, bins=20, alpha=0.7, color="#2196F3", label="Normal", density=True)
    ax.hist(anomaly_scores, bins=20, alpha=0.7, color="#F44336", label="Anomaly", density=True)

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    return save_path


def plot_precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    save_path: str,
) -> str:
    """Plot precision-recall curve with AUC annotation. Simplified for LLM vision.

    Args:
        y_true: True labels.
        scores: Anomaly scores (negated decision_function for sklearn convention).
        save_path: Path to save PNG.

    Returns:
        Path to saved file.
    """
    # Negate scores so higher = more anomalous (sklearn convention)
    precision, recall, _ = precision_recall_curve(y_true, -scores)
    pr_auc = auc(recall, precision)

    # Plain text report as image — skip the chart entirely.
    # Chart plots are slow for LLM vision; a text image is read instantly.
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.axis("off")

    # Compute best F1 from PR data
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * precision * recall / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    best_f1 = float(np.max(f1_scores))

    text = (
        f"PR Curve Summary\n"
        f"AUC = {pr_auc:.3f}\n"
        f"Best F1 = {best_f1:.3f}"
    )
    ax.text(0.5, 0.5, text, fontsize=18, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes,
            family="monospace")

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    return save_path
