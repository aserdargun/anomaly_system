"""Inference tool — predict anomalies on new data."""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from anomaly_system.config import MODELS_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def predict_new_data(
    input_path: str,
    model_path: str | None = None,
    scaler_path: str | None = None,
) -> dict:
    """Run inference on new data using a trained model and scaler.

    Args:
        input_path: Path to input CSV or parquet file.
        model_path: Path to saved model. Defaults to latest in models dir.
        scaler_path: Path to saved scaler. Defaults to tmp/scaler.joblib.

    Returns:
        Dict with output path and prediction summary.
    """
    try:
        print(f"[Step] Running inference on {input_path}")

        # Load model
        if model_path is None:
            model_path = str(MODELS_DIR / "isolation_forest.joblib")
        model = joblib.load(model_path)

        # Load scaler
        if scaler_path is None:
            scaler_path = str(OUTPUT_DIR / "tmp" / "scaler.joblib")
        scaler = joblib.load(scaler_path)

        # Load data
        path = Path(input_path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        # Remove label column if present
        label_col = None
        if "label" in df.columns:
            label_col = df["label"]
            df = df.drop(columns=["label"])

        # Preprocess
        X = scaler.transform(df.values)

        # Predict
        raw_preds = model.predict(X)
        scores = model.decision_function(X)
        predictions = (raw_preds == -1).astype(int)

        # Build output
        output_df = df.copy()
        output_df["anomaly_score"] = scores
        output_df["prediction"] = predictions
        if label_col is not None:
            output_df["true_label"] = label_col.values

        output_path = OUTPUT_DIR / "inference_results.csv"
        output_df.to_csv(output_path, index=False)

        n_anomalies = int(predictions.sum())
        print(f"[Step] Inference complete: {n_anomalies} anomalies detected out of {len(df)} samples")

        return {
            "output_path": str(output_path),
            "n_samples": len(df),
            "n_anomalies_detected": n_anomalies,
            "anomaly_ratio": round(float(predictions.mean()), 4),
        }
    except Exception as e:
        return {"error": f"Inference failed: {e}"}


def main() -> None:
    """CLI entry point for standalone inference."""
    parser = argparse.ArgumentParser(description="Run anomaly detection inference")
    parser.add_argument("--input", required=True, help="Path to input data file")
    parser.add_argument("--model", default=None, help="Path to model file")
    parser.add_argument("--scaler", default=None, help="Path to scaler file")
    args = parser.parse_args()

    result = predict_new_data(args.input, args.model, args.scaler)
    print(result)


if __name__ == "__main__":
    main()
