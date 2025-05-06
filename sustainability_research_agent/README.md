# Internet-Search and Analysis Agent (using Dual Google Gemini Models with Timeout)

## Project Goal

This project implements an agent capable of two primary functions:

1.  **Conversational Q&A:** Answering general user queries using the LangChain ReAct framework, conversation memory, and a DuckDuckGo search tool when necessary (guided by `prompt_template.txt`).
2.  **Sustainability Report Analysis:** Performing a multi-step analysis of sustainability reports for a given industry. This involves identifying key companies, searching for their reports, downloading PDFs, extracting text, summarizing individual reports, and synthesizing overall trends.

This version uses **two Google Gemini models** via `langchain-google-genai`:
*   A more powerful model (e.g., `gemini-2.5-pro-preview-03-25`) for reasoning, ReAct agent tasks, and final synthesis.
*   A faster model (e.g., `gemini-1.5-flash-latest`) specifically for the potentially repetitive summarization of individual document chunks, configured with a **request timeout** to prevent hangs during long summarization calls.

## Architecture

The agent combines two approaches and utilizes two LLM instances:

*   **LLM (Main):** `ChatGoogleGenerativeAI` instance using `GEMINI_MODEL` (e.g., `gemini-2.5-pro-preview-03-25`) used for:
    *   The ReAct Agent (general Q&A).
    *   Identifying companies in the analysis workflow.
    *   Synthesizing final trends from summaries in the analysis workflow.
*   **LLM (Summarizer):** `ChatGoogleGenerativeAI` instance using `GEMINI_SUMMARY_MODEL` (e.g., `gemini-1.5-flash-latest`) configured with `request_timeout=30` (in `agent.py`), used *only* for:
    *   The `map_reduce` steps within `load_summarize_chain`.

**Components:**
1.  **ReAct Agent (for General Q&A):**
    *   Uses the main Gemini LLM.
    *   Loads prompt instructions from `prompt_template.txt`.
    *   Uses `ConversationBufferMemory`.
    *   Has access to tools (`DuckDuckGo Search`, `Download PDF`, `Extract PDF Text`).
    *   Managed by `create_react_agent` and `AgentExecutor`.
2.  **Workflow Function (for Report Analysis):**
    *   `run_sustainability_report_analysis(industry)` in `agent.py`.
    *   Orchestrates steps:
        *   Uses the **main LLM** to identify companies.
        *   Uses search tool.
        *   Uses download/extract tools.
        *   Uses text splitter.
        *   Uses `load_summarize_chain` configured with the **summarizer LLM (with timeout)** to summarize each report.
        *   Uses the **main LLM** again to synthesize trends from summaries.
    *   Triggered by `analyze industry [Industry Name]`.

## Setup

1.  **Prerequisites:**
    *   Python 3.x installed.
    *   **Google API Key:** Required for Gemini models. Obtain from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   **Set Environment Variable:** Set `GOOGLE_API_KEY`.
        *   Linux/macOS: `export GOOGLE_API_KEY='YOUR_API_KEY'`
        *   Windows CMD: `set GOOGLE_API_KEY=YOUR_API_KEY`
        *   Windows PowerShell: `$env:GOOGLE_API_KEY='YOUR_API_KEY'`

2.  **Clone/Download:** Obtain project files.

3.  **Navigate to Project Directory:** `cd sustainability_research_agent`

4.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # or .venv\Scripts\activate
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Installs `langchain`, `langchain-community`, `langchain_text_splitters`, `langchain-google-genai`, `google-generativeai`, `duckduckgo-search`, `requests`, `pypdf`)*

## Usage

1.  Ensure `GOOGLE_API_KEY` is set.
2.  Activate virtual environment.
3.  Run the agent: `python agent.py`
4.  Interact:
    *   Ask general questions (uses main LLM).
    *   Use `analyze industry [Industry Name]` (uses main LLM for company ID/synthesis, summarizer LLM with timeout for individual report summaries).
5.  Type `quit` or `exit` to stop.

## Approach and Observations (Summarizer Timeout)

*   **Timeout:** Added a `request_timeout=30` parameter when initializing the `ChatGoogleGenerativeAI` instance used for summarization (`llm_summarizer`). This helps prevent the application from hanging indefinitely if a specific summarization call to the Gemini API takes too long. If a timeout occurs, the summarization for that specific report chunk will likely fail, and an error will be logged (if debug mode is on) or handled by the chain's error handling.

## Deliverables Checklist

*   [x] Functional Internet-Search Agent (ReAct part).
*   [x] Implemented workflow for sustainability report analysis.
*   [x] Agent includes conversation memory (for ReAct part).
*   [x] Agent loads ReAct prompt from `prompt_template.txt`.
*   [x] Uses Dual Google Gemini LLMs (main and summarizer).
*   [x] Summarizer LLM configured with a request timeout.
*   [x] Uses DuckDuckGo search.
*   [x] Includes PDF download/extraction.
*   [x] Uses LangChain for summarization/text splitting.
*   [x] Documentation (This `README.md` file).
*   [x] `requirements.txt` listing dependencies.
*   [x] `.gitignore` file.
