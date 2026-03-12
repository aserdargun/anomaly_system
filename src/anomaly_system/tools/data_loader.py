"""Data loading and preprocessing tools for the LLM agent."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from anomaly_system.config import DATA_DIR, OUTPUT_DIR, WINDOW_SIZES
from anomaly_system.tools.data_utils import (
    add_rolling_features,
    fit_scaler,
    handle_missing_values,
    save_processed_data,
    split_train_test,
    transform_with_scaler,
)

logger = logging.getLogger(__name__)


def load_and_preprocess_data(
    data_path: str,
    label_column: str = "label",
    use_feature_engineering: bool = False,
    window_sizes: list[int] | None = None,
) -> dict:
    """Load dataset, preprocess, split, and scale.

    Args:
        data_path: Path to CSV or parquet file.
        label_column: Name of the label column (0=normal, 1=anomaly).
        use_feature_engineering: Whether to add rolling features.
        window_sizes: Window sizes for rolling features.

    Returns:
        Dict with dataset info and paths to processed files.
    """
    try:
        print(f"[Step] Loading data from {data_path}")
        path = Path(data_path)

        # Load data
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        print(f"[Step] Loaded {len(df)} rows, {len(df.columns)} columns")

        # Handle missing values
        df = handle_missing_values(df)

        # Feature engineering
        if use_feature_engineering:
            ws = window_sizes or WINDOW_SIZES
            print(f"[Step] Adding rolling features with windows {ws}")
            df = add_rolling_features(df, ws, label_column)

        # Split: train on normal only, test on all
        X_train, X_test, y_train, y_test = split_train_test(df, label_column)
        print(f"[Step] Train: {len(X_train)} normal samples, Test: {len(X_test)} samples")

        # Scale
        X_train_scaled, scaler = fit_scaler(X_train)
        X_test_scaled = transform_with_scaler(X_test, scaler)

        feature_names = list(X_train.columns)

        # Save processed data
        tmp_dir = OUTPUT_DIR / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        train_path = save_processed_data(X_train_scaled, tmp_dir / "X_train.parquet", feature_names)
        test_path = save_processed_data(X_test_scaled, tmp_dir / "X_test.parquet", feature_names)
        save_processed_data(y_test.reset_index(drop=True).to_frame("label"), tmp_dir / "y_test.parquet")
        save_processed_data(y_train.reset_index(drop=True).to_frame("label"), tmp_dir / "y_train.parquet")

        scaler_path = str(tmp_dir / "scaler.joblib")
        joblib.dump(scaler, scaler_path)

        anomaly_ratio = float(y_test.sum() / len(y_test)) if len(y_test) > 0 else 0.0

        return {
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "anomaly_ratio_test": round(anomaly_ratio, 4),
            "processed_train_path": train_path,
            "processed_test_path": test_path,
            "scaler_path": scaler_path,
        }
    except Exception as e:
        return {"error": f"Data loading failed: {e}"}


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    anomaly_ratio: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Generate synthetic dataset for testing the pipeline.

    Args:
        n_samples: Total number of samples.
        n_features: Number of features.
        anomaly_ratio: Fraction of anomalous samples.
        random_state: Random seed.

    Returns:
        Dict with data path and dataset info.
    """
    try:
        print(f"[Step] Generating synthetic data: {n_samples} samples, {n_features} features")
        rng = np.random.RandomState(random_state)

        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies

        # Normal data: multivariate normal
        normal_data = rng.randn(n_normal, n_features)

        # Anomalies: shifted and scaled outliers
        anomaly_data = rng.randn(n_anomalies, n_features) * 3 + rng.choice([-4, 4], size=(n_anomalies, n_features))

        # Combine
        X = np.vstack([normal_data, anomaly_data])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

        # Shuffle
        shuffle_idx = rng.permutation(len(X))
        X = X[shuffle_idx]
        labels = labels[shuffle_idx]

        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["label"] = labels.astype(int)

        # Save
        data_path = DATA_DIR / "synthetic_data.csv"
        df.to_csv(data_path, index=False)

        print(f"[Step] Saved synthetic data to {data_path}")
        return {
            "data_path": str(data_path),
            "n_samples": n_samples,
            "n_features": n_features,
            "n_anomalies": n_anomalies,
            "anomaly_ratio": anomaly_ratio,
        }
    except Exception as e:
        return {"error": f"Synthetic data generation failed: {e}"}
