"""Main entry point for the LLM-driven anomaly detection system."""

import argparse
import asyncio
import logging
import platform
import sys
from pathlib import Path

from anomaly_system import config
from anomaly_system.llm.ollama_client import OllamaClient, OllamaNotRunning, ModelNotFound
from anomaly_system.llm.tool_registry import ToolRegistry
from anomaly_system.llm.agent import Agent
from anomaly_system.tools import register_all_tools


DEFAULT_GOAL = (
    "Run a complete anomaly detection experiment with Isolation Forest using default parameters. "
    "Generate synthetic data if no dataset exists. Train the model, evaluate with F1 score, "
    "generate all visualizations, analyze each chart image, and write a comprehensive experiment report."
)


async def run_health_check() -> None:
    """Check system readiness: Apple Silicon, Ollama, model, memory."""
    print("[Health] Checking system...")

    # Platform
    machine = platform.machine()
    system = platform.system()
    is_apple_silicon = machine == "arm64" and system == "Darwin"
    print(f"  Platform: {system} {machine} ({'Apple Silicon' if is_apple_silicon else 'NOT Apple Silicon'})")

    # Memory (macOS specific)
    if system == "Darwin":
        import subprocess
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True,
            )
            mem_bytes = int(result.stdout.strip())
            mem_gb = mem_bytes / (1024 ** 3)
            print(f"  Unified Memory: {mem_gb:.1f} GB")
        except Exception:
            print("  Unified Memory: unknown")

    # Ollama
    client = OllamaClient()
    try:
        info = await client.health_check()
        print(f"  Ollama: running")
        print(f"  Model: {info['model_name']}")
        print(f"  Metal: {'yes' if info['metal_accelerated'] else 'no'}")
        print(f"  Available models: {info['available_models']}")
        print("\n[Health] Mac Mini M4 ready!")
    except OllamaNotRunning as e:
        print(f"  Ollama: NOT RUNNING — {e}")
        print("\n[Health] Fix: brew services restart ollama")
    except ModelNotFound as e:
        print(f"  Ollama: running, but model missing — {e}")
        print(f"\n[Health] Fix: ollama pull {config.OLLAMA_MODEL}")


async def run_agent(goal: str) -> None:
    """Run the full LLM agent pipeline."""
    thinking = config.OLLAMA_THINKING
    print(f"[Agent] Initializing... (thinking: {'on' if thinking else 'off'})")

    # Health check
    client = OllamaClient(thinking=thinking)
    try:
        info = await client.health_check()
        print(f"[Agent] Ollama ready: {info['model_name']} (Metal: {info['metal_accelerated']})")
    except (OllamaNotRunning, ModelNotFound) as e:
        print(f"[Agent] ERROR: {e}")
        print("[Agent] Falling back to --pipeline mode")
        run_pipeline(None)
        return

    # Initialize registry and register all tools
    registry = ToolRegistry()
    register_all_tools(registry, client)
    print(f"[Agent] Registered {len(registry.list_tools())} tools: {registry.list_tools()}")

    # Run agent
    agent = Agent(registry, client)
    result = await agent.run(goal)

    # Print final report
    print("\n" + "=" * 80)
    print("EXPERIMENT REPORT")
    print("=" * 80)
    print(result)

    # Save report
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = config.REPORTS_DIR / f"agent_report_{timestamp}.md"
    report_path.write_text(result, encoding="utf-8")
    print(f"\n[Agent] Report saved to {report_path}")


def run_pipeline(data_path: str | None) -> None:
    """Run ML pipeline directly without LLM (fallback mode)."""
    print("[Pipeline] Running direct ML pipeline (no LLM)...")

    from anomaly_system.tools.data_loader import generate_synthetic_data, load_and_preprocess_data
    from anomaly_system.tools.isolation_forest import train_isolation_forest
    from anomaly_system.tools.evaluation import compute_metrics, generate_visualizations

    # Generate or load data
    if data_path is None:
        print("[Pipeline] No data path provided, generating synthetic data...")
        data_result = generate_synthetic_data()
        if "error" in data_result:
            print(f"[Pipeline] ERROR: {data_result['error']}")
            return
        data_path = data_result["data_path"]

    # Preprocess
    prep_result = load_and_preprocess_data(data_path)
    if "error" in prep_result:
        print(f"[Pipeline] ERROR: {prep_result['error']}")
        return
    print(f"[Pipeline] Data: {prep_result['n_train_samples']} train, {prep_result['n_test_samples']} test, {prep_result['n_features']} features")

    # Train
    train_result = train_isolation_forest()
    if "error" in train_result:
        print(f"[Pipeline] ERROR: {train_result['error']}")
        return
    print(f"[Pipeline] Model trained in {train_result['training_time_seconds']}s, {train_result['n_anomalies_detected']} anomalies detected")

    # Evaluate
    metrics = compute_metrics()
    if "error" in metrics:
        print(f"[Pipeline] ERROR: {metrics['error']}")
        return
    print(f"[Pipeline] F1={metrics['f1']}, Precision={metrics['precision']}, Recall={metrics['recall']}, ROC-AUC={metrics['roc_auc']}")

    # Visualize
    viz_result = generate_visualizations()
    if "error" in viz_result:
        print(f"[Pipeline] ERROR: {viz_result['error']}")
        return
    print(f"[Pipeline] Plots saved to outputs/plots/")

    print("\n[Pipeline] Done!")


async def run_vision_test() -> None:
    """Test vision analysis on existing plots."""
    from anomaly_system.llm.vision_analyzer import create_analyze_chart_tool

    client = OllamaClient(thinking=config.OLLAMA_THINKING)
    try:
        await client.health_check()
    except (OllamaNotRunning, ModelNotFound) as e:
        print(f"[Vision Test] ERROR: {e}")
        return

    analyze_chart = create_analyze_chart_tool(client)

    plots_dir = config.PLOTS_DIR
    chart_types = {
        "confusion_matrix.png": "confusion_matrix",
        "score_distribution.png": "score_distribution",
        "pr_curve.png": "pr_curve",
    }

    for filename, chart_type in chart_types.items():
        path = plots_dir / filename
        if path.exists():
            print(f"\n[Vision Test] Analyzing {filename}...")
            result = await analyze_chart(image_path=str(path), chart_type=chart_type)
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Analysis:\n{result['analysis']}")
        else:
            print(f"[Vision Test] {filename} not found, skipping")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-Driven Anomaly Detection (Mac Mini M4)"
    )
    parser.add_argument(
        "--agent", action="store_true", default=True,
        help="Run with LLM agent orchestration (default)",
    )
    parser.add_argument(
        "--pipeline", action="store_true",
        help="Run ML pipeline directly without LLM",
    )
    parser.add_argument(
        "--vision-test", action="store_true",
        help="Test vision analysis on existing plots",
    )
    parser.add_argument(
        "--health", action="store_true",
        help="Check Ollama, Metal, model, and memory status",
    )
    parser.add_argument(
        "--goal", type=str, default=DEFAULT_GOAL,
        help="Goal for the agent",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to input dataset",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--think", action="store_true", default=None,
        help="Enable Qwen 3.5 thinking mode (slower, more reasoning)",
    )
    thinking_group.add_argument(
        "--no-think", action="store_true", default=None,
        help="Disable Qwen 3.5 thinking mode (faster responses)",
    )
    args = parser.parse_args()

    # Resolve thinking mode
    if args.no_think:
        config.OLLAMA_THINKING = False
    elif args.think:
        config.OLLAMA_THINKING = True

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.health:
        asyncio.run(run_health_check())
    elif args.pipeline:
        run_pipeline(args.data_path)
    elif args.vision_test:
        asyncio.run(run_vision_test())
    else:
        asyncio.run(run_agent(args.goal))


if __name__ == "__main__":
    main()
