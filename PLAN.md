# Plan: Internet-Search Agent Development

## Goal

Develop an Internet-Search Agent powered by a Language Model (LLM) that utilizes an internet searching tool (like DuckDuckGo) to answer user queries, intelligently deciding when to perform a web search versus relying on internal knowledge.

## Phase 1: Setup and Foundation

1.  **Environment Setup:**
    *   Ensure Python is installed.
    *   Create a dedicated project directory (e.g., `internet_search_agent`).
    *   Set up a Python virtual environment within the project directory.
    *   Install necessary Python libraries:
        *   LLM interaction library (e.g., `langchain`, `llama-index`, or Ollama client).
        *   Web search library (e.g., `duckduckgo_search`).
    *   Install and configure a local LLM provider (e.g., Ollama) and download a suitable model (e.g., Llama 3.1 variant).

## Phase 2: Core Agent Implementation

2.  **LLM Interaction Module (`llm_interface.py`):**
    *   Handle communication with the configured LLM.
    *   Implement functions for sending prompts and receiving responses.
3.  **Search Module (`search_tool.py`):**
    *   Handle web searches using the chosen library (e.g., `duckduckgo_search`).
    *   Implement a function to take a query and return processed search results.
4.  **Agent Logic Module (`agent.py`):**
    *   **Query Handling:** Accept user input.
    *   **Search Decision:**
        *   Implement logic (basic filters + LLM assessment) to determine if a web search is necessary.
        *   Prompt example for LLM assessment: *"User query: '[user_query]'. Do you need to search the internet to answer this accurately? Respond YES or NO."*
    *   **Response Generation:**
        *   If no search: Use LLM's internal knowledge.
        *   If search: Call `search_tool`, provide results to LLM for synthesis.
        *   Prompt example for synthesis: *"Based on the following search results: [search_results], answer the user's query: '[user_query]'."*
    *   **Main Loop:** Orchestrate the query-response flow.

## Phase 3: Documentation and Deliverables

5.  **Documentation (`README.md`):**
    *   Document purpose, architecture, setup, usage, observations, and findings.
6.  **Code:** Ensure well-commented Python code.

## Proposed Architecture Diagram

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

## Clarification Points

*   **LLM Choice:** Defaulting to Ollama with a Llama 3.1 model unless specified otherwise.
*   **Search Decision:** Primarily relies on LLM self-assessment combined with basic filters.