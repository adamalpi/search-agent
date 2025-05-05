# Use absolute import assuming 'sustainability_research_agent' is in the Python path during tests
from research_agent.search_tool import (  # Updated import path
    _perform_duckduckgo_search,
    search_langchain_tool,
)


# Test the internal search function directly
def test_perform_duckduckgo_search_success(mocker):
    """Test successful search and formatting."""
    # Mock the DDGS class and its text method
    # Mock the DDGS class itself
    mock_ddgs_class = mocker.patch(
        "research_agent.search_tool.DDGS",  # Corrected mock path
    )
    # Mock the instance returned by DDGS() and its __enter__ method
    # The instance returned by __enter__ is what's assigned to 'ddgs' in the 'with' block
    mock_ddgs_instance = mock_ddgs_class.return_value.__enter__.return_value
    # Configure the 'text' method on the instance returned by __enter__
    mock_ddgs_instance.text.return_value = [
        {"title": "Result 1", "href": "http://example.com/1", "body": "Snippet 1..."},
        {"title": "Result 2", "href": "http://example.com/2", "body": "Snippet 2..."},
    ]

    query = "test query"
    result = _perform_duckduckgo_search(query, max_results=2)

    assert "Result 1" in result
    assert "http://example.com/1" in result
    assert "Snippet 1" in result
    assert "Result 2" in result
    assert "http://example.com/2" in result
    assert "Snippet 2" in result
    assert result.startswith("Search Results:")
    mock_ddgs_instance.text.assert_called_once_with(query, max_results=2)


def test_perform_duckduckgo_search_no_results(mocker):
    """Test search returning no results."""
    # Mock the DDGS class itself
    mock_ddgs_class = mocker.patch(
        "research_agent.search_tool.DDGS",  # Corrected mock path
    )
    # Mock the instance returned by DDGS() and its __enter__ method
    mock_ddgs_instance = mock_ddgs_class.return_value.__enter__.return_value
    # Configure the 'text' method
    mock_ddgs_instance.text.return_value = []  # Simulate no results

    query = "unlikely query"
    result = _perform_duckduckgo_search(query)

    assert result == "No relevant search results found."
    mock_ddgs_instance.text.assert_called_once_with(
        query, max_results=5
    )  # Default max_results


def test_perform_duckduckgo_search_error(mocker):
    """Test handling of exception during search."""
    # Mock the DDGS class itself
    mock_ddgs_class = mocker.patch(
        "research_agent.search_tool.DDGS",  # Corrected mock path
    )
    # Mock the instance returned by DDGS() and its __enter__ method
    mock_ddgs_instance = mock_ddgs_class.return_value.__enter__.return_value
    # Configure the 'text' method
    mock_ddgs_instance.text.side_effect = Exception(
        "Search engine down"
    )  # Simulate error

    query = "error query"
    result = _perform_duckduckgo_search(query)

    assert result.startswith("Error during search:")
    assert "Search engine down" in result


# Test the LangChain Tool wrapper
def test_search_langchain_tool_run(mocker):
    """Test running the search via the LangChain Tool wrapper."""
    # Mock the internal function that the tool calls
    # Mock the _run method of the tool instance directly
    mock_tool_run = mocker.patch.object(search_langchain_tool, "_run")
    mock_tool_run.return_value = "Formatted search results string"

    query = "tool test query"
    result = search_langchain_tool.run(query)  # This will now call the mocked _run

    assert result == "Formatted search results string"
    mock_tool_run.assert_called_once_with(
        query
    )  # Check if the tool's run method was called


def test_search_langchain_tool_attributes():
    """Test the tool's configured attributes."""
    assert search_langchain_tool.name == "DuckDuckGo Search"
    assert search_langchain_tool.description is not None
    assert callable(search_langchain_tool.func)
