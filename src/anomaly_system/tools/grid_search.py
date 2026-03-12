"""Grid search tool — exhaustive hyperparameter search for Isolation Forest."""

import itertools
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

from anomaly_system.config import GRID_SEARCH_PARAMS, IF_DEFAULTS, OUTPUT_DIR

logger = logging.getLogger(__name__)


def run_grid_search(param_grid: dict | None = None) -> dict:
    """Run exhaustive grid search over Isolation Forest hyperparameters.

    Args:
        param_grid: Dict of parameter lists to search. Uses config defaults if None.

    Returns:
        Dict with best params, best F1, and results path.
    """
    try:
        grid = param_grid or GRID_SEARCH_PARAMS
        tmp_dir = OUTPUT_DIR / "tmp"

        X_train = pd.read_parquet(tmp_dir / "X_train.parquet").values
        X_test = pd.read_parquet(tmp_dir / "X_test.parquet").values
        y_test = pd.read_parquet(tmp_dir / "y_test.parquet")["label"].values

        # Generate all combinations
        param_names = list(grid.keys())
        param_values = list(grid.values())
        combinations = list(itertools.product(*param_values))
        total = len(combinations)

        print(f"[Step] Grid search: {total} combinations")

        results = []
        best_f1 = -1.0
        best_params: dict = {}

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            full_params = {**IF_DEFAULTS, **params}

            try:
                start = time.time()
                model = IsolationForest(
                    n_estimators=full_params["n_estimators"],
                    contamination=full_params["contamination"],
                    max_samples=full_params["max_samples"],
                    max_features=full_params["max_features"],
                    random_state=full_params["random_state"],
                    n_jobs=-1,
                )
                model.fit(X_train)
                elapsed = time.time() - start

                raw_preds = model.predict(X_test)
                preds = (raw_preds == -1).astype(int)
                f1 = float(f1_score(y_test, preds))

                result = {
                    **{k: str(v) for k, v in params.items()},
                    "f1": round(f1, 4),
                    "training_time": round(elapsed, 4),
                }
                results.append(result)

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = params

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{total}] F1={f1:.4f} | Best so far: {best_f1:.4f}")

            except Exception as e:
                results.append({
                    **{k: str(v) for k, v in params.items()},
                    "f1": 0.0,
                    "error": str(e),
                })

        # Sort by F1
        results.sort(key=lambda x: x.get("f1", 0), reverse=True)

        # Save results
        results_dir = OUTPUT_DIR / "grid_search"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(results)
        csv_path = results_dir / "grid_search_results.csv"
        results_df.to_csv(csv_path, index=False)

        json_path = results_dir / "grid_search_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"[Step] Grid search complete. Best F1={best_f1:.4f}")
        print(f"[Step] Best params: {best_params}")

        return {
            "best_params": {k: str(v) for k, v in best_params.items()},
            "best_f1": round(best_f1, 4),
            "total_combinations": total,
            "results_path": str(csv_path),
        }
    except Exception as e:
        return {"error": f"Grid search failed: {e}"}
