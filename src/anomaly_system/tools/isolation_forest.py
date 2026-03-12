"""Isolation Forest model training and prediction tools."""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from anomaly_system.config import IF_DEFAULTS, MODELS_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def train_isolation_forest(
    n_estimators: int = IF_DEFAULTS["n_estimators"],
    contamination: str | float = IF_DEFAULTS["contamination"],
    max_samples: str | int = IF_DEFAULTS["max_samples"],
    max_features: float = IF_DEFAULTS["max_features"],
    random_state: int = IF_DEFAULTS["random_state"],
) -> dict:
    """Train an Isolation Forest model on preprocessed data.

    Args:
        n_estimators: Number of isolation trees.
        contamination: Expected proportion of outliers (or 'auto').
        max_samples: Number of samples to draw for each tree.
        max_features: Number of features to draw for each tree.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with model path, training time, and predictions info.
    """
    try:
        tmp_dir = OUTPUT_DIR / "tmp"
        train_path = tmp_dir / "X_train.parquet"
        test_path = tmp_dir / "X_test.parquet"
        y_test_path = tmp_dir / "y_test.parquet"

        if not train_path.exists():
            return {"error": "Processed training data not found. Run load_and_preprocess_data first."}

        print(f"[Step] Training Isolation Forest: n_estimators={n_estimators}, contamination={contamination}")

        X_train = pd.read_parquet(train_path).values
        X_test = pd.read_parquet(test_path).values

        # Train
        start_time = time.time()
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # Use all M4 cores
        )
        model.fit(X_train)
        training_time = time.time() - start_time

        print(f"[Step] Training completed in {training_time:.2f}s")

        # Predict on test data
        raw_preds = model.predict(X_test)
        scores = model.decision_function(X_test)

        # Convert sklearn convention: -1=anomaly, 1=normal → 0=normal, 1=anomaly
        predictions = (raw_preds == -1).astype(int)

        # Save model
        model_path = MODELS_DIR / "isolation_forest.joblib"
        joblib.dump(model, model_path)

        # Save predictions and scores
        pred_df = pd.DataFrame({
            "prediction": predictions,
            "anomaly_score": scores,
        })
        pred_path = tmp_dir / "predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)

        n_detected = int(predictions.sum())
        print(f"[Step] Detected {n_detected} anomalies in test set")

        return {
            "model_path": str(model_path),
            "training_time_seconds": round(training_time, 4),
            "n_estimators": n_estimators,
            "contamination": str(contamination),
            "max_samples": str(max_samples),
            "max_features": max_features,
            "n_test_samples": len(predictions),
            "n_anomalies_detected": n_detected,
            "predictions_path": str(pred_path),
        }
    except Exception as e:
        return {"error": f"Training failed: {e}"}


def predict_with_model(model_path: str, data_path: str) -> dict:
    """Load a saved model and predict on new data.

    Args:
        model_path: Path to saved joblib model.
        data_path: Path to data file (parquet or CSV).

    Returns:
        Dict with predictions path and summary stats.
    """
    try:
        print(f"[Step] Predicting with model {model_path}")

        model = joblib.load(model_path)

        path = Path(data_path)
        if path.suffix == ".parquet":
            data = pd.read_parquet(path)
        else:
            data = pd.read_csv(path)

        raw_preds = model.predict(data.values)
        scores = model.decision_function(data.values)
        predictions = (raw_preds == -1).astype(int)

        tmp_dir = OUTPUT_DIR / "tmp"
        pred_df = pd.DataFrame({
            "prediction": predictions,
            "anomaly_score": scores,
        })
        pred_path = tmp_dir / "predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)

        return {
            "predictions_path": str(pred_path),
            "n_samples": len(predictions),
            "n_anomalies_detected": int(predictions.sum()),
            "anomaly_ratio": round(float(predictions.mean()), 4),
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
