"""ML tools — register all tools with the LLM tool registry."""

from anomaly_system.llm.ollama_client import OllamaClient
from anomaly_system.llm.tool_registry import ToolRegistry
from anomaly_system.llm.vision_analyzer import create_analyze_chart_tool
from anomaly_system.llm.report_generator import create_generate_report_tool, update_log
from anomaly_system.tools.data_loader import generate_synthetic_data, load_and_preprocess_data
from anomaly_system.tools.evaluation import compute_metrics, generate_visualizations
from anomaly_system.tools.grid_search import run_grid_search
from anomaly_system.tools.inference import predict_new_data
from anomaly_system.tools.isolation_forest import predict_with_model, train_isolation_forest


def register_all_tools(registry: ToolRegistry, client: OllamaClient) -> ToolRegistry:
    """Register all ML and LLM tools with the registry.

    Args:
        registry: ToolRegistry instance to register tools with.
        client: OllamaClient for vision and report tools.

    Returns:
        The registry with all tools registered.
    """
    # --- Data tools ---

    registry.register(
        name="generate_synthetic_data",
        description="Generate a synthetic dataset for anomaly detection testing. Creates normal data from a multivariate normal distribution and injects anomalies as shifted/scaled outliers.",
        parameters={
            "type": "object",
            "properties": {
                "n_samples": {"type": "integer", "description": "Total number of samples (default 1000)"},
                "n_features": {"type": "integer", "description": "Number of features (default 10)"},
                "anomaly_ratio": {"type": "number", "description": "Fraction of anomalous samples (default 0.05)"},
                "random_state": {"type": "integer", "description": "Random seed (default 42)"},
            },
            "required": [],
        },
    )(generate_synthetic_data)

    registry.register(
        name="load_and_preprocess_data",
        description="Load a dataset from CSV/parquet, handle missing values, split into train (normal only) and test (all), scale features with StandardScaler, and optionally add rolling features.",
        parameters={
            "type": "object",
            "properties": {
                "data_path": {"type": "string", "description": "Path to the dataset file (CSV or parquet)"},
                "label_column": {"type": "string", "description": "Name of the label column (default 'label')"},
                "use_feature_engineering": {"type": "boolean", "description": "Whether to add rolling window features (default false)"},
                "window_sizes": {"type": "array", "items": {"type": "integer"}, "description": "Window sizes for rolling features (default [10, 30, 60])"},
            },
            "required": ["data_path"],
        },
    )(load_and_preprocess_data)

    # --- Model tools ---

    registry.register(
        name="train_isolation_forest",
        description="Train an Isolation Forest model on preprocessed data. Uses all M4 CPU cores. Saves the model and generates predictions on test data.",
        parameters={
            "type": "object",
            "properties": {
                "n_estimators": {"type": "integer", "description": "Number of isolation trees (default 100)"},
                "contamination": {"type": "string", "description": "Expected outlier fraction or 'auto' (default 'auto')"},
                "max_samples": {"type": "string", "description": "Samples per tree: 'auto' or integer as string (default 'auto')"},
                "max_features": {"type": "number", "description": "Features per tree as fraction (default 1.0)"},
                "random_state": {"type": "integer", "description": "Random seed (default 42)"},
            },
            "required": [],
        },
    )(train_isolation_forest)

    registry.register(
        name="predict_with_model",
        description="Load a saved Isolation Forest model and predict anomalies on a dataset.",
        parameters={
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to the saved model file"},
                "data_path": {"type": "string", "description": "Path to the data file to predict on"},
            },
            "required": ["model_path", "data_path"],
        },
    )(predict_with_model)

    # --- Evaluation tools ---

    registry.register(
        name="compute_metrics",
        description="Compute F1, Precision, Recall, and ROC-AUC metrics from the latest model predictions.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )(compute_metrics)

    registry.register(
        name="generate_visualizations",
        description="Generate confusion matrix heatmap, anomaly score distribution, and precision-recall curve plots as PNG images.",
        parameters={
            "type": "object",
            "properties": {
                "output_dir": {"type": "string", "description": "Optional directory to save plots (default: outputs/plots/)"},
            },
            "required": [],
        },
    )(generate_visualizations)

    # --- Grid search ---

    registry.register(
        name="run_grid_search",
        description="Run exhaustive grid search over Isolation Forest hyperparameters (n_estimators, contamination, max_samples). Returns best parameters and F1 score.",
        parameters={
            "type": "object",
            "properties": {
                "param_grid": {
                    "type": "object",
                    "description": "Dict of parameter lists to search. Uses defaults if not provided.",
                },
            },
            "required": [],
        },
    )(run_grid_search)

    # --- Inference ---

    registry.register(
        name="predict_new_data",
        description="Run inference on new data using a trained model and scaler. Outputs CSV with original data, anomaly scores, and predictions.",
        parameters={
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Path to input data file"},
                "model_path": {"type": "string", "description": "Path to saved model (optional)"},
                "scaler_path": {"type": "string", "description": "Path to saved scaler (optional)"},
            },
            "required": ["input_path"],
        },
    )(predict_new_data)

    # --- Vision analysis ---

    analyze_chart = create_analyze_chart_tool(client)
    registry.register(
        name="analyze_chart",
        description="Analyze a chart/plot image using LLM vision. Sends the image to Qwen 3.5 with a chart-specific prompt and returns textual analysis.",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to the chart PNG file"},
                "chart_type": {
                    "type": "string",
                    "description": "Type of chart: 'confusion_matrix', 'score_distribution', or 'pr_curve'",
                    "enum": ["confusion_matrix", "score_distribution", "pr_curve"],
                },
            },
            "required": ["image_path", "chart_type"],
        },
    )(analyze_chart)

    # --- Report generation ---

    generate_report = create_generate_report_tool(client)
    registry.register(
        name="generate_report",
        description="Generate a structured markdown experiment report using the LLM. Collects metrics and visual analyses, then produces a comprehensive report.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )(generate_report)

    registry.register(
        name="update_log",
        description="Append the experiment report to the project log file (log.md).",
        parameters={
            "type": "object",
            "properties": {
                "report_text": {"type": "string", "description": "Markdown report text to append to the log"},
            },
            "required": ["report_text"],
        },
    )(update_log)

    return registry
