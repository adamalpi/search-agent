import asyncio
import logging
import re

from logging_config import setup_logging

# Import only what's needed: model names for printing and the unified graph builder
from research_agent.agent import (
    GEMINI_MODEL,
    GEMINI_SUMMARY_MODEL,
)
from research_agent.graph_builder import (
    build_unified_graph,
)

# Optional: Import os and shutil for cleanup if uncommented later

setup_logging()

try:
    unified_graph_app = build_unified_graph()
except Exception as e:
    logging.error(f"Failed to build the unified graph: {e}", exc_info=True)
    print("FATAL: Could not build the unified graph. Exiting.")
    exit(1)


async def main():
    """Runs the main interaction loop using the unified graph."""
    print("--- Unified Search & Analysis Agent (LangGraph) ---")
    print(f"Using main LLM model: {GEMINI_MODEL}")
    print(f"Using summarization LLM model: {GEMINI_SUMMARY_MODEL}")
    print("Enter a general query, or type 'analyze industry [Industry Name]' to run the report analysis.")
    print("Type 'quit' or 'exit' to stop.")

    # Simple history management for the CLI interaction (graph handles internal memory)

    while True:
        user_input = input("\nUser Query: ")
        input_lower = user_input.strip().lower()

        if input_lower in ["quit", "exit"]:
            print("Exiting agent.")
            break
        if not user_input.strip():
            continue

        inputs = {}
        analysis_match = re.match(r"analyze industry\s+(.+)", input_lower)
        if analysis_match:
            industry_name = analysis_match.group(1).strip()
            print(f"\n--- Starting Unified Graph (Analysis Path) for: {industry_name} ---")
            inputs = {"industry": industry_name}
        else:
            print(f"\n--- Starting Unified Graph (Agent Path) for: {user_input} ---")
            inputs = {
                "input_query": user_input,
                "messages": [],
            }

        try:
            final_state = await unified_graph_app.ainvoke(inputs, {"recursion_limit": 10})

            print("\n--- Graph Execution Complete ---")

            if final_state.get("error_message"):
                print(f"\nError: {final_state['error_message']}")
            elif final_state.get("agent_response"):
                agent_response = final_state["agent_response"]
                print("\nAgent Response:")
                print(agent_response)
            elif final_state.get("synthesis_result"):
                analysis_result = final_state["synthesis_result"]
                print("\nAnalysis Result:")
                print(analysis_result)
            else:
                print("\nGraph finished, but no standard output (agent_response/synthesis_result/error_message) found.")
                print("Final State:", final_state)

        except Exception as e:
            logging.error(
                f"Error invoking unified graph for input '{inputs}': {e}",
                exc_info=True,
            )
            print("\nError: An unexpected error occurred while running the unified graph. Check logs.")

    # Optional: Clean up downloaded files on exit


if __name__ == "__main__":
    asyncio.run(main())
