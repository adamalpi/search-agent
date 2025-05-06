import logging

from duckduckgo_search import DDGS
from langchain.tools import Tool

logger = logging.getLogger(__name__)

# --- Configuration ---
MAX_SEARCH_RESULTS = 5

# --- Core Search Functionality ---


def _perform_duckduckgo_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """
    Performs a web search using DuckDuckGo and returns formatted results as a string.
    This is the underlying function used by the LangChain tool.

    Args:
        query: The search query string.
        max_results: The maximum number of search results to return.

    Returns:
        A formatted string containing the search results, or a message indicating no results.
    """
    logger.info(f"Performing DuckDuckGo search for query: '{query}' (max_results={max_results})")
    results_list = []
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            if search_results:
                results_list = list(search_results)[:max_results]
                logger.info(f"Found {len(results_list)} search results.")
            else:
                logger.warning(f"No search results found for query: '{query}'")
                return "No relevant search results found."
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for query '{query}': {e}")
        return f"Error during search: {e}"

    if not results_list:
        return "No relevant search results found."

    formatted_string = "Search Results:\n\n"
    for i, result in enumerate(results_list, 1):
        title = result.get("title", "N/A")
        href = result.get("href", "N/A")
        body = result.get("body", "N/A")
        formatted_string += f"{i}. Title: {title}\n"
        formatted_string += f"   URL: {href}\n"
        formatted_string += f"   Snippet: {body}\n\n"

    return formatted_string.strip()


# --- LangChain Tool Definition ---

# Note: The function passed to Tool should ideally only accept a single string argument (the query).
# We use the internal _perform_duckduckgo_search function.
search_langchain_tool = Tool(
    name="DuckDuckGo Search",
    func=_perform_duckduckgo_search,
    description="Useful for when you need to answer questions about current events, specific facts, or information not found in your internal knowledge base. Input should be a search query.",  # noqa: E501
)

# --- Example Usage (Moved to tests/test_search_tool.py) ---
