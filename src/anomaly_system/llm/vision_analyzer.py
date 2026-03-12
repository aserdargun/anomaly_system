"""Vision analysis — send chart images to Qwen 3.5 for interpretation."""

import logging

from anomaly_system.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

ANALYSIS_PROMPTS = {
    "confusion_matrix": (
        "Analyze this confusion matrix heatmap for an anomaly detection model. Report:\n"
        "1. True Positive, True Negative, False Positive, False Negative counts\n"
        "2. Which error type dominates (false positives or false negatives)?\n"
        "3. Overall assessment of the model's detection capability"
    ),
    "score_distribution": (
        "Analyze this anomaly score distribution plot. The blue histogram shows normal "
        "samples and the red shows anomalies. Report:\n"
        "1. Is there clear separation between the two distributions?\n"
        "2. How much overlap exists?\n"
        "3. What does this suggest about the model's discrimination ability?"
    ),
    "pr_curve": (
        "Analyze this Precision-Recall curve for an anomaly detection model. Report:\n"
        "1. What is the approximate area under the curve?\n"
        "2. At what point does precision start dropping significantly?\n"
        "3. What is the approximate best F1 point on the curve?"
    ),
}


def create_analyze_chart_tool(client: OllamaClient):
    """Create the analyze_chart function bound to an Ollama client.

    Args:
        client: OllamaClient instance for vision calls.

    Returns:
        The analyze_chart function ready for tool registration.
    """
    async def analyze_chart(image_path: str, chart_type: str) -> dict:
        """Analyze a chart image using LLM vision.

        Args:
            image_path: Path to the chart PNG file.
            chart_type: One of 'confusion_matrix', 'score_distribution', 'pr_curve'.

        Returns:
            Dict with analysis text.
        """
        if chart_type not in ANALYSIS_PROMPTS:
            return {"error": f"Unknown chart_type: {chart_type}. Use one of: {list(ANALYSIS_PROMPTS.keys())}"}

        prompt = ANALYSIS_PROMPTS[chart_type]
        print(f"  [Vision] Analyzing {chart_type} from {image_path}")

        try:
            analysis = await client.chat_with_vision(prompt, image_path)
            return {"analysis": analysis, "chart_type": chart_type, "image_path": image_path}
        except Exception as e:
            return {"error": f"Vision analysis failed: {e}"}

    return analyze_chart
