"""Visualization functions — headless matplotlib on macOS."""

import matplotlib
matplotlib.use("Agg")  # Must be before pyplot import — headless rendering on macOS

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

from anomaly_system.config import PLOT_DPI, PLOT_FIGSIZE


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

    # Counts and percentages
    total = cm.sum()
    annot = np.array([[f"{v}\n({v/total*100:.1f}%)" for v in row] for row in cm])

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Isolation Forest", fontsize=14)
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

    ax.hist(normal_scores, bins=50, alpha=0.6, color="blue", label="Normal", density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, color="red", label="Anomaly", density=True)

    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Anomaly Score Distribution — Isolation Forest", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    return save_path


def plot_precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    save_path: str,
) -> str:
    """Plot precision-recall curve with F1 iso-lines and AUC annotation.

    Args:
        y_true: True labels.
        scores: Anomaly scores (negated decision_function for sklearn convention).
        save_path: Path to save PNG.

    Returns:
        Path to saved file.
    """
    # Negate scores so higher = more anomalous (sklearn convention)
    precision, recall, thresholds = precision_recall_curve(y_true, -scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    # F1 iso-lines
    f1_values = [0.2, 0.4, 0.6, 0.8]
    for f1 in f1_values:
        x = np.linspace(0.01, 1, 100)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = f1 * x / (2 * x - f1)
        valid = np.isfinite(y) & (y >= 0) & (y <= 1)
        ax.plot(x[valid], y[valid], "--", color="gray", alpha=0.3)
        if valid.any():
            idx = valid.nonzero()[0][-1]
            ax.annotate(f"F1={f1}", xy=(x[idx], y[idx]), fontsize=8, color="gray")

    ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR Curve (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Isolation Forest", fontsize=14)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    return save_path
