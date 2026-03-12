"""Data utility functions — scaling, splitting, missing value handling."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using forward fill then drop remaining NaNs.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with no missing values.
    """
    df = df.ffill()
    df = df.dropna()
    return df


def split_train_test(
    df: pd.DataFrame,
    label_column: str = "label",
    test_ratio: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data: train on normal only, test on all.

    Args:
        df: Input DataFrame with features and label.
        label_column: Name of the label column (0=normal, 1=anomaly).
        test_ratio: Fraction of normal data to include in test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    features = df.drop(columns=[label_column])
    labels = df[label_column]

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    normal_data = features[normal_mask]
    normal_labels = labels[normal_mask]

    # Split normal data into train/test
    rng = np.random.RandomState(random_state)
    n_normal = len(normal_data)
    n_test_normal = int(n_normal * test_ratio)
    indices = rng.permutation(n_normal)

    test_normal_idx = normal_data.index[indices[:n_test_normal]]
    train_normal_idx = normal_data.index[indices[n_test_normal:]]

    X_train = features.loc[train_normal_idx]
    y_train = labels.loc[train_normal_idx]

    # Test set: remaining normal + all anomalies
    X_test = pd.concat([features.loc[test_normal_idx], features[anomaly_mask]])
    y_test = pd.concat([labels.loc[test_normal_idx], labels[anomaly_mask]])

    return X_train, X_test, y_train, y_test


def fit_scaler(X_train: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Fit StandardScaler on training data.

    Args:
        X_train: Training feature matrix.

    Returns:
        Tuple of (scaled training data, fitted scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    return X_scaled, scaler


def transform_with_scaler(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform data with a fitted scaler.

    Args:
        X: Feature matrix to transform.
        scaler: Fitted StandardScaler.

    Returns:
        Scaled numpy array.
    """
    return scaler.transform(X)


def add_rolling_features(
    df: pd.DataFrame,
    window_sizes: list[int],
    label_column: str = "label",
) -> pd.DataFrame:
    """Add rolling mean and std features for temporal context.

    Args:
        df: Input DataFrame.
        window_sizes: List of window sizes for rolling statistics.
        label_column: Label column to exclude from feature engineering.

    Returns:
        DataFrame with additional rolling features.
    """
    feature_cols = [c for c in df.columns if c != label_column]
    new_features = {}

    for col in feature_cols:
        for w in window_sizes:
            new_features[f"{col}_rolling_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            new_features[f"{col}_rolling_std_{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)

    new_df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    return new_df


def save_processed_data(
    data: np.ndarray | pd.DataFrame,
    path: Path,
    columns: list[str] | None = None,
) -> str:
    """Save processed data as parquet.

    Args:
        data: Data to save.
        path: Output file path.
        columns: Column names if data is a numpy array.

    Returns:
        String path to saved file.
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=columns)
    else:
        df = data
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return str(path)
