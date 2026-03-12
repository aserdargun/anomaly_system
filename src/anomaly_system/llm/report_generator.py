"""Report generation — LLM writes structured experiment reports."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from anomaly_system.config import REPORTS_DIR
from anomaly_system.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

REPORT_PROMPT_TEMPLATE = """\
Generate a structured experiment report in markdown based on the following data.

Metrics: {metrics_json}

Visual Analysis:
- Confusion Matrix: {cm_analysis}
- Score Distribution: {sd_analysis}
- PR Curve: {pr_analysis}

Parameters: {params_json}
Hardware: Mac Mini M4, Metal GPU, unified memory

Write the report with these sections:
1. Experiment Summary (one paragraph)
2. Results Table (F1, Precision, Recall, ROC-AUC)
3. Visual Analysis (key observations from each chart)
4. Conclusions and Next Steps\
"""


# Module-level state for collecting results across tool calls
_report_state: dict = {
    "metrics": None,
    "visual_analyses": {},
    "params": None,
}


def set_report_metrics(metrics: dict) -> None:
    """Store metrics for report generation."""
    _report_state["metrics"] = metrics


def set_report_visual_analysis(chart_type: str, analysis: str) -> None:
    """Store a visual analysis for report generation."""
    _report_state["visual_analyses"][chart_type] = analysis


def set_report_params(params: dict) -> None:
    """Store model parameters for report generation."""
    _report_state["params"] = params


def create_generate_report_tool(client: OllamaClient):
    """Create the generate_report function bound to an Ollama client.

    Args:
        client: OllamaClient instance for LLM calls.

    Returns:
        The generate_report function ready for tool registration.
    """
    async def generate_report() -> dict:
        """Generate a structured experiment report using the LLM.

        Returns:
            Dict with report_text and report_path.
        """
        metrics = _report_state.get("metrics") or {}
        analyses = _report_state.get("visual_analyses") or {}
        params = _report_state.get("params") or {}

        prompt = REPORT_PROMPT_TEMPLATE.format(
            metrics_json=json.dumps(metrics, indent=2, default=str),
            cm_analysis=analyses.get("confusion_matrix", "Not available"),
            sd_analysis=analyses.get("score_distribution", "Not available"),
            pr_analysis=analyses.get("pr_curve", "Not available"),
            params_json=json.dumps(params, indent=2, default=str),
        )

        print("[Report] Generating experiment report via LLM...")
        try:
            response = await client.chat(
                messages=[
                    {"role": "system", "content": "You are a technical report writer. Write clear, concise experiment reports in markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
            report_text = response["message"]["content"]
        except Exception as e:
            # Fallback: generate a basic report without LLM
            report_text = _fallback_report(metrics, analyses, params)
            logger.warning("LLM report generation failed, using fallback: %s", e)

        # Save report
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"experiment_{timestamp}.md"
        report_path.write_text(report_text, encoding="utf-8")

        print(f"[Report] Saved to {report_path}")
        return {"report_text": report_text, "report_path": str(report_path)}

    return generate_report


def _fallback_report(metrics: dict, analyses: dict, params: dict) -> str:
    """Generate a basic report without LLM assistance."""
    lines = [
        "# Experiment Report",
        f"\nDate: {datetime.now(timezone.utc).isoformat()}",
        "\n## Results",
        f"- F1 Score: {metrics.get('f1', 'N/A')}",
        f"- Precision: {metrics.get('precision', 'N/A')}",
        f"- Recall: {metrics.get('recall', 'N/A')}",
        f"- ROC-AUC: {metrics.get('roc_auc', 'N/A')}",
        "\n## Parameters",
        f"```json\n{json.dumps(params, indent=2, default=str)}\n```",
        "\n## Visual Analysis",
    ]
    for chart_type, analysis in analyses.items():
        lines.append(f"\n### {chart_type}\n{analysis}")
    return "\n".join(lines)


def update_log(report_text: str) -> dict:
    """Append experiment entry to the project log.

    Args:
        report_text: Markdown report text to append.

    Returns:
        Dict with success status and log path.
    """
    log_path = Path(__file__).parent.parent.parent.parent.parent / "log.md"
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        entry = f"\n\n---\n\n## Experiment — {timestamp}\n\n{report_text}\n"

        if log_path.exists():
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry)
        else:
            log_path.write_text(f"# Experiment Log\n{entry}", encoding="utf-8")

        print(f"[Log] Updated {log_path}")
        return {"success": True, "log_path": str(log_path)}
    except Exception as e:
        return {"error": f"Failed to update log: {e}"}
