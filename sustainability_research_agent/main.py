import re
import logging

# Import necessary components initialized in agent.py
from agent import (
    react_agent_executor,
    # run_sustainability_report_analysis, # Removed import as function moved to graph
    GEMINI_MODEL,
    GEMINI_SUMMARY_MODEL,
    # DOWNLOAD_DIR removed as it's handled within file_tools now
)
from graph_builder import build_graph  # Import the graph builder
# Optional: Import os and shutil for cleanup if uncommented later
# import os
# import shutil

# Configure logging for main script (optional, if agent.py doesn't cover it)
# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Build the graph application
try:
    analysis_graph_app = build_graph()
except Exception as e:
    logging.error(f"Failed to build the analysis graph: {e}", exc_info=True)
    print("FATAL: Could not build the analysis graph. Exiting.")
    exit(1)


def main():
    """Runs the main interaction loop for the agent."""
    print("--- Internet-Search Agent (with LangGraph Analysis) ---")
    print(f"Using main LLM model: {GEMINI_MODEL}")
    print(f"Using summarization LLM model: {GEMINI_SUMMARY_MODEL}")
    print(
        "Enter a general query, or type 'analyze industry [Industry Name]' to run the report analysis."
    )
    print("Type 'quit' or 'exit' to stop.")

    while True:
        user_input = input("\nUser Query: ")
        input_lower = user_input.strip().lower()

        if input_lower in ["quit", "exit"]:
            print("Exiting agent.")
            break
        if not user_input.strip():
            continue

        # Check if the user wants to run the specific analysis workflow
        analysis_match = re.match(r"analyze industry\s+(.+)", input_lower)
        if analysis_match:
            industry_name = analysis_match.group(1).strip()
            print(f"\n--- Starting Analysis Graph for: {industry_name} ---")
            inputs = {"industry": industry_name}
            try:
                # Invoke the graph
                # Note: Streaming or async invoke might be better for long tasks in a real UI
                final_state = analysis_graph_app.invoke(inputs)
                print("\n--- Analysis Complete ---")
                # Display result or error from the final state
                analysis_result = final_state.get(
                    "synthesis_result"
                ) or final_state.get(
                    "error_message",
                    "Graph finished, but no result/error found in state.",
                )
                print("\nAnalysis Result:")
                print(analysis_result)

            except Exception as e:
                logging.error(
                    f"Error invoking analysis graph for '{industry_name}': {e}",
                    exc_info=True,
                )
                print(
                    "\nError: An unexpected error occurred while running the analysis graph. Check logs."
                )
        else:
            # Fallback to the general ReAct agent
            logging.info(f"Invoking ReAct agent with query: '{user_input}'")
            try:
                response = react_agent_executor.invoke({"input": user_input})
                agent_response = response.get(
                    "output", "Agent did not provide a final answer."
                )
                print("\nAgent Response:")
                print(agent_response)
            except Exception as e:
                logging.error(
                    f"Error during ReAct agent execution for query '{user_input}': {e}",
                    exc_info=True,
                )
                print(
                    "\nError: An unexpected error occurred while processing your query. Check logs for details."
                )

    # Optional: Clean up downloaded files on exit
    # try:
    #     if os.path.exists(DOWNLOAD_DIR):
    #         print(f"\nCleaning up download directory: {DOWNLOAD_DIR}")
    #         shutil.rmtree(DOWNLOAD_DIR)
    #         print(f"Download directory cleaned up.")
    # except Exception as e:
    #     print(f"Error cleaning up download directory {DOWNLOAD_DIR}: {e}")


if __name__ == "__main__":
    main()
