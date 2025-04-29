import ollama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Consider making the model name configurable, e.g., via environment variable or config file
DEFAULT_OLLAMA_MODEL = "llama3.1" # Or another suitable model installed in Ollama
DEFAULT_OLLAMA_MODEL = "deepseek-r1:8b" # Or another suitable model installed in Ollama

# --- Core LLM Interaction ---

def get_llm_response(prompt: str, model: str = DEFAULT_OLLAMA_MODEL) -> str:
    """
    Sends a prompt to the configured Ollama LLM and returns the response content.

    Args:
        prompt: The input prompt string for the LLM.
        model: The name of the Ollama model to use.

    Returns:
        The content of the LLM's response as a string.
        Returns an error message string if communication fails.
    """
    logging.info(f"Sending prompt to Ollama model '{model}':\n{prompt}")
    try:
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ])
        response_content = response['message']['content']
        logging.info(f"Received response from Ollama model '{model}':\n{response_content}")
        return response_content
    except Exception as e:
        logging.error(f"Error communicating with Ollama model '{model}': {e}")
        return f"Error: Could not get response from LLM. Details: {e}"

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    test_prompt = "Explain the concept of sustainability in simple terms."
    print(f"Testing LLM interface with model: {DEFAULT_OLLAMA_MODEL}")
    print(f"Prompt: {test_prompt}")
    response = get_llm_response(test_prompt)
    print("\nResponse:")
    print(response)

    test_prompt_error = "Tell me about today's news." # Example that might require search
    print(f"\nTesting LLM interface with potentially knowledge-limited prompt:")
    print(f"Prompt: {test_prompt_error}")
    response = get_llm_response(test_prompt_error)
    print("\nResponse:")
    print(response)