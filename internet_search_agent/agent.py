import logging
from llm_interface import get_llm_response, DEFAULT_OLLAMA_MODEL
from search_tool import perform_search, format_search_results_for_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Simple greetings/phrases that generally don't require a search
SIMPLE_QUERIES = {
    "hi", "hello", "how are you", "how are you?", "thanks", "thank you",
    "ok", "okay", "bye", "goodbye"
}

# --- Agent Logic ---

def should_search(query: str, model: str = DEFAULT_OLLAMA_MODEL) -> bool:
    """
    Determines whether a web search is necessary to answer the query.

    Args:
        query: The user's input query.
        model: The Ollama model to use for the assessment.

    Returns:
        True if a search is deemed necessary, False otherwise.
    """
    # 1. Basic Filtering
    normalized_query = query.strip().lower()
    if normalized_query in SIMPLE_QUERIES:
        logging.info(f"Query '{query}' identified as simple, skipping search.")
        return False

    # 2. LLM Assessment
    # Ask the LLM if it needs external information.
    # This prompt needs to be carefully crafted for the specific LLM.
    assessment_prompt = f"""User query: "{query}"

Does answering this query accurately require searching the internet for current or specific information beyond general knowledge?
Respond with only "YES" or "NO"."""

    logging.info(f"Asking LLM ({model}) if search is needed for query: '{query}'")
    try:
        response = get_llm_response(assessment_prompt, model=model)
        decision = response.strip().upper()
        logging.info(f"LLM assessment response: '{decision}'")

        if "YES" in decision: # Check for "YES" substring for robustness
             logging.info("LLM indicates search is needed.")
             return True
        elif "NO" in decision: # Check for "NO" substring
             logging.info("LLM indicates search is not needed.")
             return False
        else:
            logging.warning(f"LLM assessment returned ambiguous response: '{response}'. Defaulting to search.")
            # Default to searching if the LLM's response isn't clear YES/NO
            return True

    except Exception as e:
        logging.error(f"Error during LLM search assessment: {e}. Defaulting to search.")
        # Default to searching if the assessment fails
        return True


def process_query(query: str, model: str = DEFAULT_OLLAMA_MODEL) -> str:
    """
    Processes a user query, decides whether to search, performs search if needed,
    and generates a final response using the LLM.

    Args:
        query: The user's input query.
        model: The Ollama model to use.

    Returns:
        A string containing the agent's final response.
    """
    logging.info(f"Processing query: '{query}'")

    if should_search(query, model=model):
        logging.info("Proceeding with web search.")
        search_results = perform_search(query)
        formatted_results = format_search_results_for_llm(search_results)

        # Prompt LLM to synthesize search results
        synthesis_prompt = f"""Based on the following search results:

{formatted_results}

Please provide a comprehensive answer to the user's original query: "{query}"

Synthesize the information from the search results into a clear and concise response. If the search results are irrelevant or insufficient, state that you couldn't find specific information from the web search."""

        logging.info("Asking LLM to synthesize search results.")
        final_response = get_llm_response(synthesis_prompt, model=model)

    else:
        logging.info("Answering query using LLM's internal knowledge.")
        # Prompt LLM to answer directly
        direct_answer_prompt = f"""Please answer the following user query: "{query}" """
        final_response = get_llm_response(direct_answer_prompt, model=model)

    return final_response


# --- Main Interaction Loop ---
if __name__ == "__main__":
    print("--- Internet-Search Agent ---")
    print(f"Using LLM model: {DEFAULT_OLLAMA_MODEL}")
    print("Enter your query below (type 'quit' or 'exit' to stop).")

    while True:
        user_input = input("\nUser Query: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting agent.")
            break
        if not user_input.strip():
            continue

        response = process_query(user_input)
        print("\nAgent Response:")
        print(response)
