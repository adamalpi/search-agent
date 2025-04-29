# Internet-Search Agent

## Project Goal

This project implements an Internet-Search Agent as described in Task 1 of the Siemens Analytics Lab tasks. The agent utilizes a Language Model (LLM) via Ollama and integrates the DuckDuckGo search engine to answer user queries. It intelligently determines whether to perform a web search based on the query or rely on the LLM's internal knowledge.

## Architecture

The agent consists of three main Python modules:

1.  **`llm_interface.py`**: Handles all communication with the configured Ollama LLM. It sends prompts and retrieves generated responses.
2.  **`search_tool.py`**: Performs web searches using the `duckduckgo-search` library. It fetches search results (title, URL, snippet) for a given query.
3.  **`agent.py`**: Contains the core agent logic. It orchestrates the workflow:
    *   Receives user queries.
    *   Uses basic filtering and an LLM assessment (`should_search` function) to decide if a web search is necessary.
    *   If a search is needed, it calls `search_tool.py` and then prompts `llm_interface.py` to synthesize the results.
    *   If no search is needed, it prompts `llm_interface.py` to answer directly.
    *   Provides a simple command-line interface for interaction.

```mermaid
graph TD
    subgraph User Interaction
        A[User Input Query]
    end

    subgraph Agent Core (`agent.py`)
        B[Query Handler]
        C{Search Needed?}
        D[Response Generator]
    end

    subgraph Tools
        E[LLM Interface (`llm_interface.py`)] --- F((Ollama LLM))
        G[Search Tool (`search_tool.py`)] --- H((DuckDuckGo))
    end

    subgraph Output
        I[Formatted Response]
    end

    A --> B;
    B --> C;
    C -- No --> E;
    C -- Yes --> G;
    G --> D;
    E --> D;
    D --> I;
```

## Setup

1.  **Prerequisites:**
    *   Python 3.x installed.
    *   [Ollama](https://ollama.com/) installed and running.
    *   An Ollama model downloaded (e.g., `ollama pull llama3.1`). The agent defaults to `llama3.1`, but this can be changed in `llm_interface.py`.

2.  **Clone/Download:** Obtain the project files.

3.  **Create Virtual Environment (Recommended):**
    ```bash
    cd internet_search_agent
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created containing `ollama` and `duckduckgo-search`)*

## Usage

1.  Ensure Ollama is running in the background.
2.  Activate the virtual environment (if created): `source .venv/bin/activate`.
3.  Run the agent script:
    ```bash
    python agent.py
    ```
4.  Enter your queries at the prompt. Type `quit` or `exit` to stop the agent.

## Approach and Observations

*   **LLM Choice:** Ollama was chosen as suggested for utilizing open-source models locally. `llama3.1` is used as the default model, assuming it's available.
*   **Search Engine:** DuckDuckGo was used via the `duckduckgo-search` library as it doesn't require an API key.
*   **Search Decision Logic:** The core challenge is determining *when* to search. The current approach uses:
    *   A simple list (`SIMPLE_QUERIES`) to quickly filter out greetings/trivial inputs.
    *   An LLM self-assessment prompt. This relies heavily on the LLM's ability to understand the prompt and accurately judge its knowledge limits for the given query. The prompt is designed to elicit a clear "YES" or "NO".
    *   **Potential Improvement:** The LLM assessment could be made more robust. Different prompt strategies or fine-tuning a model specifically for this classification task could yield better results. Error handling defaults to performing a search if the LLM response is ambiguous or fails, which is a safer default for an information-retrieval task.
*   **Result Synthesis:** The LLM is prompted to synthesize the search results and answer the original query, rather than just presenting raw links. This addresses the requirement for human-readable output.
*   **Limitations:**
    *   The quality of the search decision depends heavily on the chosen LLM and the effectiveness of the assessment prompt.
    *   Search result quality from DuckDuckGo can vary.
    *   The agent currently lacks conversation history/memory. Each query is treated independently.

## Deliverables Checklist

*   [x] Fully functional Internet-Search Agent (`agent.py`, `llm_interface.py`, `search_tool.py`)
*   [x] Documentation of the findings and approach (This `README.md` file)