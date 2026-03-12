"""Core agent loop — LLM orchestrates ML pipeline via tool calls."""

import json
import logging

from anomaly_system.config import AGENT_MAX_ITERATIONS
from anomaly_system.llm.ollama_client import OllamaClient
from anomaly_system.llm.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an ML experiment assistant running on a Mac Mini M4. You run anomaly detection experiments using the tools available to you.

Workflow:
1. Load and preprocess the data
2. Train an Isolation Forest model
3. Evaluate the model (compute F1, precision, recall, ROC-AUC)
4. Generate visualizations (confusion matrix, score distribution, PR curve)
5. Analyze each visualization image using your vision capability
6. Write a comprehensive experiment report

Always complete all steps. After generating visualizations, you MUST analyze each image before writing the report. The report should include both numeric metrics and your visual observations.

Return your final experiment report in markdown format.\
"""


class Agent:
    """LLM agent that drives the anomaly detection pipeline."""

    def __init__(
        self,
        registry: ToolRegistry,
        client: OllamaClient,
        max_iterations: int = AGENT_MAX_ITERATIONS,
    ) -> None:
        """Initialize agent with tool registry and Ollama client."""
        self.registry = registry
        self.client = client
        self.messages: list[dict] = []
        self.max_iterations = max_iterations

    async def run(self, goal: str) -> str:
        """Run agent loop until LLM produces a final answer.

        Args:
            goal: High-level objective for the agent.

        Returns:
            Final text response from the LLM.
        """
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]

        tool_defs = self.registry.get_definitions()
        print(f"[Agent] Starting with {len(tool_defs)} tools available")
        print(f"[Agent] Goal: {goal[:200]}")

        for i in range(self.max_iterations):
            print(f"\n[Agent] Iteration {i + 1}/{self.max_iterations}")

            response = await self.client.chat(
                messages=self.messages,
                tools=tool_defs,
            )

            message = response["message"]
            self.messages.append(message)

            # If LLM made tool calls, execute them
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    name = tool_call["function"]["name"]
                    args = tool_call["function"]["arguments"]
                    print(f"  [Tool Call] {name}({json.dumps(args)[:200]})")

                    result = await self.registry.execute(name, args)
                    result_str = json.dumps(result, default=str)
                    print(f"  [Tool Result] {name} -> {result_str[:200]}")

                    self.messages.append({
                        "role": "tool",
                        "content": result_str,
                    })
            else:
                # No tool calls = final answer
                content = message.get("content", "")
                print(f"\n[Agent] Final answer received ({len(content)} chars)")
                return content

        return "Agent reached max iterations without final answer."
