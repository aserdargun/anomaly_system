"""Evaluation tools — compute metrics and generate visualizations."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from anomaly_system.config import OUTPUT_DIR, PLOTS_DIR
from anomaly_system.tools.visualization import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_score_distribution,
)

logger = logging.getLogger(__name__)


def compute_metrics() -> dict:
    """Compute classification metrics from latest predictions.

    Returns:
        Dict with f1, precision, recall, roc_auc, and support counts.
    """
    try:
        tmp_dir = OUTPUT_DIR / "tmp"
        y_test = pd.read_parquet(tmp_dir / "y_test.parquet")["label"].values
        preds = pd.read_parquet(tmp_dir / "predictions.parquet")

        y_pred = preds["prediction"].values
        scores = preds["anomaly_score"].values

        f1 = float(f1_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))

        # ROC-AUC uses negated scores (lower decision_function = more anomalous)
        try:
            roc_auc = float(roc_auc_score(y_test, -scores))
        except ValueError:
            roc_auc = 0.0

        support_normal = int((y_test == 0).sum())
        support_anomaly = int((y_test == 1).sum())

        print(f"[Step] Metrics: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, ROC-AUC={roc_auc:.4f}")

        return {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "roc_auc": round(roc_auc, 4),
            "support_normal": support_normal,
            "support_anomaly": support_anomaly,
        }
    except Exception as e:
        return {"error": f"Metrics computation failed: {e}"}


def generate_visualizations(output_dir: str | None = None) -> dict:
    """Generate all evaluation plots.

    Args:
        output_dir: Optional override for output directory.

    Returns:
        Dict with paths to generated plot files.
    """
    try:
        tmp_dir = OUTPUT_DIR / "tmp"
        y_test = pd.read_parquet(tmp_dir / "y_test.parquet")["label"].values
        preds = pd.read_parquet(tmp_dir / "predictions.parquet")

        y_pred = preds["prediction"].values
        scores = preds["anomaly_score"].values

        plots_dir = PLOTS_DIR if output_dir is None else __import__("pathlib").Path(output_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        print("[Step] Generating visualizations...")

        cm_path = plot_confusion_matrix(
            y_test, y_pred, str(plots_dir / "confusion_matrix.png")
        )
        print(f"  [Plot] Confusion matrix saved to {cm_path}")

        sd_path = plot_score_distribution(
            y_test, scores, str(plots_dir / "score_distribution.png")
        )
        print(f"  [Plot] Score distribution saved to {sd_path}")

        pr_path = plot_precision_recall_curve(
            y_test, scores, str(plots_dir / "pr_curve.png")
        )
        print(f"  [Plot] PR curve saved to {pr_path}")

        return {
            "confusion_matrix_path": cm_path,
            "score_distribution_path": sd_path,
            "pr_curve_path": pr_path,
        }
    except Exception as e:
        return {"error": f"Visualization generation failed: {e}"}
