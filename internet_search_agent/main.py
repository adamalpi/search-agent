import re
import logging
# Import necessary components initialized in agent.py
from agent import (
    react_agent_executor,
    run_sustainability_report_analysis,
    GEMINI_MODEL,
    GEMINI_SUMMARY_MODEL
    # DOWNLOAD_DIR removed as it's handled within file_tools now
)
# Optional: Import os and shutil for cleanup if uncommented later
# import os
# import shutil

# Configure logging for main script (optional, if agent.py doesn't cover it)
# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Runs the main interaction loop for the agent."""
    print("--- Internet-Search Agent ---")
    print(f"Using main LLM model: {GEMINI_MODEL}")
    print(f"Using summarization LLM model: {GEMINI_SUMMARY_MODEL}")
    print("Enter a general query, or type 'analyze industry [Industry Name]' to run the report analysis.")
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
            try:
                analysis_result = run_sustainability_report_analysis(industry_name)
                print("\nAnalysis Result:")
                print(analysis_result)
            except Exception as e:
                logging.error(f"Error during sustainability analysis for '{industry_name}': {e}", exc_info=True)
                print(f"\nError: An unexpected error occurred during the analysis workflow. Check logs for details.")

        else:
            # Fallback to the general ReAct agent
            logging.info(f"Invoking ReAct agent with query: '{user_input}'")
            try:
                response = react_agent_executor.invoke({"input": user_input})
                agent_response = response.get('output', "Agent did not provide a final answer.")
                print("\nAgent Response:")
                print(agent_response)
            except Exception as e:
                logging.error(f"Error during ReAct agent execution for query '{user_input}': {e}", exc_info=True)
                print(f"\nError: An unexpected error occurred while processing your query. Check logs for details.")

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