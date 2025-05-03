import logging
import os  # Import os for environment variables

# from langchain_community.llms import Ollama # Remove Ollama import
from langchain_google_genai import ChatGoogleGenerativeAI  # Import Gemini
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import (
    ChatPromptTemplate,
)  # Remove PromptTemplate, no longer needed here
from langchain.memory import ConversationBufferMemory

# Remove summarization/splitting imports, moved to graph_builder
# from langchain.chains.summarize import load_summarize_chain
# from langchain_text_splitters import RecursiveCharacterTextSplitter
import langchain  # Import langchain base for debug setting
from search_tool import search_langchain_tool
from file_tools import download_pdf_tool, extract_pdf_text_tool

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("pypdf").setLevel(logging.ERROR)  # Silence pypdf info logs

# Enable LangChain debug mode for detailed logs
langchain.debug = True
logging.info("LangChain global debug mode enabled.")

# --- Configuration ---
# OLLAMA_MODEL = "deepseek-r1:8b" # Commented out Ollama model
GEMINI_MODEL = (
    "gemini-2.5-pro-preview-03-25"  # Main model for reasoning, synthesis, ReAct
)
GEMINI_SUMMARY_MODEL = (
    "gemini-2.5-flash-preview-04-17"  # Faster model specifically for summarization
)
PROMPT_FILE = "prompts/prompt_template.txt"  # Updated path

# --- LangChain Setup ---

# 1. Initialize the LLM (Gemini)
# Ensure the GOOGLE_API_KEY environment variable is set
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print(
        "Please set the GOOGLE_API_KEY environment variable before running the script."
    )
    print("Example (Linux/macOS): export GOOGLE_API_KEY='your_api_key_here'")
    print("Example (Windows CMD): set GOOGLE_API_KEY=your_api_key_here")
    print("Example (Windows PowerShell): $env:GOOGLE_API_KEY='your_api_key_here'")
    exit(1)

try:
    # Use ChatGoogleGenerativeAI for chat-based interaction
    # Initialize main LLM (for reasoning, synthesis, ReAct)
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=google_api_key)
    logging.info(f"Successfully initialized main Google Gemini model: {GEMINI_MODEL}")

    # Initialize separate LLM for summarization with a timeout
    llm_summarizer = ChatGoogleGenerativeAI(
        model=GEMINI_SUMMARY_MODEL,
        google_api_key=google_api_key,
        request_timeout=30,  # Timeout in seconds for each summarization API call
    )
    logging.info(
        f"Successfully initialized summarization Google Gemini model: {GEMINI_SUMMARY_MODEL} with 120s timeout"
    )

    # Test connections (optional)
    # llm.invoke("Hi")
    # llm_summarizer.invoke("Hi")
except Exception as e:
    logging.error(
        f"Failed to initialize Google Gemini model '{GEMINI_MODEL}'. Error: {e}"
    )
    print(
        f"Error: Could not initialize Google Gemini model '{GEMINI_MODEL}'. Check API key and configuration."
    )
    exit(1)

# 2. Define the tools for the ReAct agent (general Q&A)
# Tools remain the same
react_tools = [search_langchain_tool, download_pdf_tool, extract_pdf_text_tool]

# 3. Load the Custom ReAct Prompt Template from file
try:
    with open(PROMPT_FILE, "r") as f:
        template_string = f.read()
    react_prompt = ChatPromptTemplate.from_template(template_string)
    logging.info(f"Successfully loaded and created prompt template from {PROMPT_FILE}.")
except Exception as e:
    logging.error(f"Failed to create prompt template from {PROMPT_FILE}: {e}")
    print("Error: Could not create the agent prompt template.")
    exit(1)

# 4. Initialize Conversation Memory (for ReAct agent)
react_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
logging.info("Initialized ConversationBufferMemory for ReAct agent.")

# 5. Create the ReAct Agent (for general Q&A)
try:
    react_agent = create_react_agent(llm, react_tools, react_prompt)
    logging.info("Successfully created ReAct agent with custom prompt and memory.")
except Exception as e:
    logging.error(f"Failed to create ReAct agent: {e}")
    print("Error: Could not create the LangChain agent.")
    exit(1)

# 6. Create the ReAct Agent Executor (for general Q&A)
react_agent_executor = AgentExecutor(
    agent=react_agent,
    tools=react_tools,
    memory=react_memory,
    verbose=True,
    handle_parsing_errors=True,
)
logging.info("Successfully created ReAct Agent Executor with memory.")

# --- Sustainability Report Analysis Workflow (Moved to graph_builder.py) ---
# The setup code above initializes variables (llm, llm_summarizer, react_agent_executor, etc.)
# Some of these (llm, llm_summarizer) are imported by graph_builder.py
# The react_agent_executor is imported by main.py for general queries.
# The compiled graph app from graph_builder.py is imported by main.py for the analysis task.
