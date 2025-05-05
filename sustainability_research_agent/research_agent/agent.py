import logging
import os

# Remove summarization/splitting imports, moved to graph_builder
import langchain
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from research_agent.file_tools import (
    download_pdf_tool,
    extract_pdf_text_tool,
)
from research_agent.search_tool import (
    search_langchain_tool,
)

# Removed logging configuration - Handled centrally by setup_logging() in entry points (main.py/api_server.py)
# logging.basicConfig(...)
# logging.getLogger("langchain")...
# logging.getLogger("pypdf")...


# Enable LangChain debug mode for detailed logs
langchain.debug = True
if langchain.debug:
    logging.info("LangChain global debug mode enabled.")

# --- Configuration ---
# OLLAMA_MODEL = "deepseek-r1:8b" # Commented out Ollama model

GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"
GEMINI_SUMMARY_MODEL = "gemini-2.5-flash-preview-04-17"
PROMPT_FILE = "prompts/prompt_template.txt"

TOOLS = [search_langchain_tool, download_pdf_tool, extract_pdf_text_tool]

# --- LangChain Setup ---


def initialize_gemini():
    """Initialize the client to connect with the remote LLM."""

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set the GOOGLE_API_KEY environment variable before running the script.")
        exit(1)

    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=google_api_key)
        logging.info(f"Successfully initialized main Google Gemini model: {GEMINI_MODEL}")

        llm_summarizer = ChatGoogleGenerativeAI(
            model=GEMINI_SUMMARY_MODEL,
            google_api_key=google_api_key,
            request_timeout=30,  # Timeout in seconds for each API call
        )
        logging.info(
            f"Successfully initialized summarization Google Gemini model: {GEMINI_SUMMARY_MODEL} with 120s timeout"
        )

    except Exception as e:
        logging.error(f"Failed to initialize Google Gemini model '{GEMINI_MODEL}'. Error: {e}")
        print(f"Error: Could not initialize Google Gemini model '{GEMINI_MODEL}'. Check API key and configuration.")
        exit(1)

    return llm, llm_summarizer


def load_prompt():
    """Load the Custom Prompt Template from file using an absolute path."""
    try:
        script_dir = os.path.dirname(__file__)
        absolute_prompt_path = os.path.join(os.path.dirname(script_dir), PROMPT_FILE)

        with open(absolute_prompt_path, "r") as f:
            template_string = f.read()
        react_prompt = ChatPromptTemplate.from_template(template_string)
        logging.info(f"Successfully loaded and created prompt template from {absolute_prompt_path}.")
    except Exception as e:
        # Use absolute_prompt_path in error message if it was defined
        prompt_path_for_error = absolute_prompt_path if "absolute_prompt_path" in locals() else PROMPT_FILE
        logging.error(f"Failed to create prompt template from {prompt_path_for_error}: {e}")
        print(f"Error: Could not create the agent prompt template from {prompt_path_for_error}.")
        exit(1)

    return react_prompt


def initialize_agent(llm):
    prompt_template = load_prompt()

    react_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    logging.info("Initialized ConversationBufferMemory for ReAct agent.")

    try:
        react_agent = create_react_agent(llm, TOOLS, prompt_template)
        logging.info("Successfully created ReAct agent with custom prompt and memory.")
    except Exception as e:
        logging.error(f"Failed to create ReAct agent: {e}")
        print("Error: Could not create the LangChain agent.")
        exit(1)

    react_agent_executor = AgentExecutor(
        agent=react_agent,
        tools=TOOLS,
        memory=react_memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    logging.info("Successfully created ReAct Agent Executor with memory.")

    return react_agent_executor


# llm, llm_summarizer = initialize_gemini() # Moved initialization to where it's needed
# react_agent_executor = initialize_agent(llm) # Moved initialization to where it's needed
