import re
import logging

# Import only what's needed: model names for printing and the unified graph builder
from research_agent.agent import (
    GEMINI_MODEL,
    GEMINI_SUMMARY_MODEL,
)
from research_agent.graph_builder import (
    build_unified_graph,
)  # Import the unified graph builder
# Optional: Import os and shutil for cleanup if uncommented later
# Configure logging for main script (optional, if agent.py doesn't cover it)
# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the unified graph once
try:
    unified_graph_app = build_unified_graph()
except Exception as e:
    logging.error(f"Failed to build the unified graph: {e}", exc_info=True)
    print("FATAL: Could not build the unified graph. Exiting.")
    exit(1)


def main():
    """Runs the main interaction loop using the unified graph."""
    print("--- Unified Search & Analysis Agent (LangGraph) ---")
    print(f"Using main LLM model: {GEMINI_MODEL}")
    print(f"Using summarization LLM model: {GEMINI_SUMMARY_MODEL}")
    print(
        "Enter a general query, or type 'analyze industry [Industry Name]' to run the report analysis."
    )
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

        # Prepare inputs for the unified graph
        inputs = {}
        analysis_match = re.match(r"analyze industry\s+(.+)", input_lower)
        if analysis_match:
            industry_name = analysis_match.group(1).strip()
            print(
                f"\n--- Starting Unified Graph (Analysis Path) for: {industry_name} ---"
            )
            inputs = {"industry": industry_name}
            # Clear history for analysis tasks? Or keep it? Let's clear for now.
        else:
            print(f"\n--- Starting Unified Graph (Agent Path) for: {user_input} ---")
            # Pass the current query and the message history
            # The graph expects LangChain message objects in the 'messages' key
            # For simplicity here, we'll just pass the query. The graph's run_basic_agent
            # node currently creates its own memory, but ideally, we'd pass history.
            # Let's adapt to pass history (as dicts for simplicity, graph node converts)
            # inputs = {"input_query": user_input, "messages": chat_history}
            # Reverting to simpler input for now, as graph node handles memory internally
            inputs = {
                "input_query": user_input,
                "messages": [],
            }  # Pass empty history for now

        # Invoke the unified graph
        try:
            # Use stream for better feedback (optional)
            # final_state = None
            # for event in unified_graph_app.stream(inputs, {"recursion_limit": 10}):
            #     print(f"Event: {event}")
            #     # Extract final state from the stream if possible
            #     # This depends on the structure of your stream events
            #     # Example: if "__end__" in event: final_state = event["__end__"]

            # Or just invoke for the final result
            final_state = unified_graph_app.invoke(
                inputs, {"recursion_limit": 10}
            )  # Add recursion limit

            print("\n--- Graph Execution Complete ---")

            # Extract and display the result
            if final_state.get("error_message"):
                print(f"\nError: {final_state['error_message']}")
            elif final_state.get("agent_response"):  # Check for agent response first
                agent_response = final_state["agent_response"]
                print("\nAgent Response:")
                print(agent_response)
                # Update CLI history (simplified)
                # chat_history.append({"type": "human", "content": user_input})
                # chat_history.append({"type": "ai", "content": agent_response})
            elif final_state.get("synthesis_result"):
                analysis_result = final_state["synthesis_result"]
                print("\nAnalysis Result:")
                print(analysis_result)
            else:
                print(
                    "\nGraph finished, but no standard output (agent_response/synthesis_result/error_message) found."
                )
                print("Final State:", final_state)  # Print raw state for debugging

        except Exception as e:
            logging.error(
                f"Error invoking unified graph for input '{inputs}': {e}",
                exc_info=True,
            )
            print(
                "\nError: An unexpected error occurred while running the unified graph. Check logs."
            )

    # Optional: Clean up downloaded files on exit


if __name__ == "__main__":
    main()
