from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import Runnable
from logging_config import setup_logging  # Import setup
from research_agent.graph_builder import build_unified_graph

# logging.basicConfig(...) # Removed - Handled centrally


@pytest.fixture(autouse=True)
def mock_llm_init(mocker):
    """Automatically mock the LLM initialization and setup logging for all tests."""
    setup_logging()  # Setup logging here for tests
    mock_init = mocker.patch("research_agent.graph_builder.initialize_gemini")

    class MockRunnableLLM(MagicMock, Runnable):
        def invoke(self, *args, **kwargs):
            return MagicMock()

        def batch(self, *args, **kwargs):
            return [MagicMock()]

        def stream(self, *args, **kwargs):
            yield MagicMock()

    mock_llm_instance = MockRunnableLLM()
    mock_summarizer_instance = MockRunnableLLM()
    mock_init.return_value = (mock_llm_instance, mock_summarizer_instance)

    yield mock_init


@pytest.mark.integration
def test_graph_build_and_invoke():
    """
    Tests if the graph can be built and invoked with a sample industry
    without raising an exception. Does not validate the content deeply
    due to the complexity and external dependencies (LLMs, search).
    """

    print("\n--- Building Graph for Test ---")
    try:
        graph_app = build_unified_graph()
    except Exception as e:
        pytest.fail(f"Failed to build the graph: {e}")

    test_industry = "renewable energy"
    print(f"\n--- Invoking Graph for Test Industry: {test_industry} ---")
    inputs = {"industry": test_industry}

    final_state = None
    try:
        # Stream events (optional, useful for debugging)
        print("Streaming events:")
        for event in graph_app.stream(inputs, {"recursion_limit": 5}):  # Add recursion limit
            print(f"Event: {event}")

        # Invoke to get final state (redundant if streaming captures it, but good practice)
        print("\nInvoking graph to get final state...")
        final_state = graph_app.invoke(inputs, {"recursion_limit": 5})  # Add recursion limit

        print("\n--- Final State Received ---")
        print(final_state)

        # Basic assertions: Check if the process completed and produced some result or a known error
        assert final_state is not None
        assert isinstance(final_state, dict)
        # Check if either a result or an error message is present (one should ideally be)
        assert final_state.get("synthesis_result") or final_state.get(
            "error_message"
        ), "Graph finished but neither synthesis_result nor error_message was found in the final state."

        if final_state.get("error_message"):
            print(f"Graph execution completed with an error message: {final_state['error_message']}")
        else:
            result_len = len(final_state.get("synthesis_result", ""))
            print(f"Graph execution completed successfully. Synthesis result length: {result_len}")
            assert isinstance(final_state.get("synthesis_result"), str)
            assert len(final_state.get("synthesis_result", "")) > 0

    except Exception as e:
        pytest.fail(f"Error invoking graph during test: {e}")
