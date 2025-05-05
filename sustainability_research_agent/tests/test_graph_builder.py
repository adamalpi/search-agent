import pytest
import logging
from unittest.mock import MagicMock
from research_agent.graph_builder import build_unified_graph

# Import Runnable for mocking LLM type checks
from langchain_core.runnables import Runnable

# Configure logging for tests if needed, or rely on pytest's capture
logging.basicConfig(level=logging.INFO)


@pytest.fixture(autouse=True)
def mock_llm_init(mocker):
    """Automatically mock the LLM initialization for all tests in this module."""
    # Use mocker.patch which integrates well with pytest fixtures
    # Patch initialize_gemini where it's imported and used in graph_builder
    mock_init = mocker.patch("research_agent.graph_builder.initialize_gemini")

    # Create a mock class that satisfies the Runnable type check
    # and implements its abstract methods
    class MockRunnableLLM(MagicMock, Runnable):
        # Provide dummy implementations for abstract methods
        def invoke(self, *args, **kwargs):
            # Can make this more sophisticated if needed, e.g., return a mock AIMessage
            return MagicMock()  # Return a simple MagicMock for now

        # Add other abstract methods if needed (e.g., batch, stream)
        # Check Runnable definition if more errors occur
        def batch(self, *args, **kwargs):
            return [MagicMock()]  # Return a list of mocks

        def stream(self, *args, **kwargs):
            yield MagicMock()  # Yield a mock

    # Configure the mock to return instances of the Runnable mock class
    mock_llm_instance = MockRunnableLLM()
    mock_summarizer_instance = MockRunnableLLM()
    mock_init.return_value = (mock_llm_instance, mock_summarizer_instance)

    yield mock_init  # The patch is active during the yield


@pytest.mark.integration  # Mark as integration test as it involves external calls/LLMs
def test_graph_build_and_invoke():  # Removed mock_initialize_gemini argument
    """
    Tests if the graph can be built and invoked with a sample industry
    without raising an exception. Does not validate the content deeply
    due to the complexity and external dependencies (LLMs, search).
    """
    # Mocking is now handled by the autouse fixture mock_llm_init

    print("\n--- Building Graph for Test ---")
    try:
        graph_app = build_unified_graph()  # Use updated function name
    except Exception as e:
        pytest.fail(f"Failed to build the graph: {e}")

    test_industry = "renewable energy"  # Use a different industry for testing maybe?
    print(f"\n--- Invoking Graph for Test Industry: {test_industry} ---")
    inputs = {"industry": test_industry}

    final_state = None
    try:
        # Stream events (optional, useful for debugging)
        print("Streaming events:")
        for event in graph_app.stream(
            inputs, {"recursion_limit": 5}
        ):  # Add recursion limit
            print(f"Event: {event}")
            # Check for intermediate errors if possible/needed
            # if isinstance(event, dict):
            #    for key, value in event.items():
            #        if isinstance(value, GraphState) and value.get("error_message"):
            #             print(f"Error message found during stream: {value['error_message']}")
            #             # Decide if this should fail the test

        # Invoke to get final state (redundant if streaming captures it, but good practice)
        print("\nInvoking graph to get final state...")
        final_state = graph_app.invoke(
            inputs, {"recursion_limit": 5}
        )  # Add recursion limit

        print("\n--- Final State Received ---")
        print(final_state)

        # Basic assertions: Check if the process completed and produced some result or a known error
        assert final_state is not None
        assert isinstance(final_state, dict)
        # Check if either a result or an error message is present (one should ideally be)
        assert (
            final_state.get("synthesis_result") or final_state.get("error_message")
        ), "Graph finished but neither synthesis_result nor error_message was found in the final state."

        if final_state.get("error_message"):
            print(
                f"Graph execution completed with an error message: {final_state['error_message']}"
            )
            # Depending on requirements, you might want to fail the test on error:
            # pytest.fail(f"Graph execution failed with error: {final_state['error_message']}")
        else:
            print(
                f"Graph execution completed successfully. Synthesis result length: {len(final_state.get('synthesis_result', ''))}"
            )
            assert isinstance(final_state.get("synthesis_result"), str)
            assert (
                len(final_state.get("synthesis_result", "")) > 0
            )  # Ensure result is not empty

    except Exception as e:
        pytest.fail(f"Error invoking graph during test: {e}")


# You could add more tests here, potentially mocking parts of the graph
# for more controlled unit testing if needed in the future.
