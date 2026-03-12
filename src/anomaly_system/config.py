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
OLLAMA_TIMEOUT = 120  # seconds for text chat
OLLAMA_VISION_TIMEOUT = 300  # seconds for vision
VISION_MAX_IMAGE_WIDTH = 224  # resize for vision — fewer pixels = fewer tokens = faster
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

# Visualization — kept small for fast LLM vision processing
# Qwen 3.5 vision: patch_size=16, spatial_merge=2 → tokens ≈ (W/32)*(H/32)
# At DPI=72, figsize=(5,3.5) → 360x252px → ~28 tokens (fast)
PLOT_DPI = 72
PLOT_FIGSIZE = (5, 3.5)
MATPLOTLIB_BACKEND = "Agg"  # Use Agg for headless; "macosx" for interactive on M4
