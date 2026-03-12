"""Configuration for anomaly detection system on Mac Mini M4."""

from pathlib import Path
import platform

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "anomaly_system" / "data"
OUTPUT_DIR = PROJECT_ROOT / "src" / "anomaly_system" / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure output directories exist
for _dir in [DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Ollama (Metal-accelerated on Mac Mini M4)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:latest"
OLLAMA_TIMEOUT = 180  # seconds — M4 Metal is fast, but vision calls can take longer
AGENT_MAX_ITERATIONS = 20
OLLAMA_THINKING = True  # Qwen 3.5 thinking mode — disable with --no-think for speed

# Platform check
IS_APPLE_SILICON = platform.machine() == "arm64" and platform.system() == "Darwin"

# Isolation Forest defaults
IF_DEFAULTS = {
    "n_estimators": 100,
    "contamination": "auto",
    "max_samples": "auto",
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,  # Use all M4 performance + efficiency cores
}

# Grid search ranges
GRID_SEARCH_PARAMS = {
    "n_estimators": [50, 100, 200, 500],
    "contamination": [0.01, 0.05, 0.1, "auto"],
    "max_samples": ["auto", 256, 512],
}

# Feature engineering
WINDOW_SIZES = [10, 30, 60]

# Visualization
PLOT_DPI = 150
PLOT_FIGSIZE = (10, 7)
MATPLOTLIB_BACKEND = "Agg"  # Use Agg for headless; "macosx" for interactive on M4
