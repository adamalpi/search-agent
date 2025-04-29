from duckduckgo_search import DDGS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MAX_SEARCH_RESULTS = 5 # Number of search results to retrieve

# --- Search Functionality ---

def perform_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """
    Performs a web search using DuckDuckGo and returns formatted results.

    Args:
        query: The search query string.
        max_results: The maximum number of search results to return.

    Returns:
        A list of dictionaries, where each dictionary represents a search result
        containing 'title', 'href' (URL), and 'body' (snippet).
        Returns an empty list if the search fails or yields no results.
    """
    logging.info(f"Performing DuckDuckGo search for query: '{query}' (max_results={max_results})")
    results = []
    try:
        with DDGS() as ddgs:
            # Using ddgs.text which is simpler and often sufficient for snippets
            search_results = ddgs.text(query, max_results=max_results)
            if search_results:
                # Ensure we only take up to max_results, even if the library returns more
                results = list(search_results)[:max_results]
                logging.info(f"Found {len(results)} search results.")
                # Format results for consistency (optional, as ddgs.text already returns dicts)
                # formatted_results = [{'title': r.get('title'), 'href': r.get('href'), 'body': r.get('body')} for r in results]
                # return formatted_results
                return results # Return the raw results from ddgs.text
            else:
                logging.warning(f"No search results found for query: '{query}'")
                return []
    except Exception as e:
        logging.error(f"Error during DuckDuckGo search for query '{query}': {e}")
        return [] # Return empty list on error

def format_search_results_for_llm(results: list[dict]) -> str:
    """
    Formats a list of search result dictionaries into a single string suitable for an LLM prompt.

    Args:
        results: A list of search result dictionaries (from perform_search).

    Returns:
        A formatted string containing the search results, or a message indicating no results.
    """
    if not results:
        return "No search results found."

    formatted_string = "Search Results:\n\n"
    for i, result in enumerate(results, 1):
        title = result.get('title', 'N/A')
        href = result.get('href', 'N/A')
        body = result.get('body', 'N/A')
        formatted_string += f"{i}. Title: {title}\n"
        formatted_string += f"   URL: {href}\n"
        formatted_string += f"   Snippet: {body}\n\n"

    return formatted_string.strip()


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    test_query = "Siemens sustainability report 2024"
    print(f"Testing search tool with query: '{test_query}'")
    search_results = perform_search(test_query)

    if search_results:
        print(f"\nRaw Results ({len(search_results)}):")
        for res in search_results:
            print(res)

        print("\nFormatted Results for LLM:")
        formatted = format_search_results_for_llm(search_results)
        print(formatted)
    else:
        print("\nNo search results obtained.")

    test_query_none = "asdlfkjasdlfkjasdlfkj" # Unlikely to yield results
    print(f"\nTesting search tool with query likely returning no results: '{test_query_none}'")
    search_results_none = perform_search(test_query_none)
    formatted_none = format_search_results_for_llm(search_results_none)
    print(formatted_none)