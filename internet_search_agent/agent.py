import logging
import re
import os # Import os for environment variables
# from langchain_community.llms import Ollama # Remove Ollama import
from langchain_google_genai import ChatGoogleGenerativeAI # Import Gemini
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, PromptTemplate # Add PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain # For summarization
from langchain_text_splitters import RecursiveCharacterTextSplitter # For splitting docs
import langchain # Import langchain base for debug setting
from search_tool import search_langchain_tool
from file_tools import download_pdf_tool, extract_pdf_text_tool # Remove DOWNLOAD_DIR import

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("pypdf").setLevel(logging.ERROR) # Silence pypdf info logs

# Enable LangChain debug mode for detailed logs
langchain.debug = True
logging.info("LangChain global debug mode enabled.")

# --- Configuration ---
# OLLAMA_MODEL = "deepseek-r1:8b" # Commented out Ollama model
GEMINI_MODEL = "gemini-2.5-pro-preview-03-25" # Main model for reasoning, synthesis, ReAct
GEMINI_SUMMARY_MODEL = "gemini-2.5-flash-preview-04-17" # Faster model specifically for summarization
PROMPT_FILE = "internet_search_agent/prompt_template.txt"

# --- LangChain Setup ---

# 1. Initialize the LLM (Gemini)
# Ensure the GOOGLE_API_KEY environment variable is set
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the GOOGLE_API_KEY environment variable before running the script.")
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
        request_timeout=30 # Timeout in seconds for each summarization API call
    )
    logging.info(f"Successfully initialized summarization Google Gemini model: {GEMINI_SUMMARY_MODEL} with 120s timeout")

    # Test connections (optional)
    # llm.invoke("Hi")
    # llm_summarizer.invoke("Hi")
except Exception as e:
    logging.error(f"Failed to initialize Google Gemini model '{GEMINI_MODEL}'. Error: {e}")
    print(f"Error: Could not initialize Google Gemini model '{GEMINI_MODEL}'. Check API key and configuration.")
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
    print(f"Error: Could not create the agent prompt template.")
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
    print(f"Error: Could not create the LangChain agent.")
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

# --- Sustainability Report Analysis Workflow ---

# Setup for summarization
# Using map_reduce, good for summarizing multiple docs independently then combining
# Load map and combine prompts for summarization chain
try:
    with open("internet_search_agent/map_prompt.txt", "r") as f:
        map_prompt_template = f.read()
    map_prompt = PromptTemplate.from_template(map_prompt_template)

    with open("internet_search_agent/combine_prompt.txt", "r") as f:
        combine_prompt_template = f.read()
    combine_prompt = PromptTemplate.from_template(combine_prompt_template)

    summarize_chain = load_summarize_chain(
        llm=llm_summarizer,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True # Use the dedicated summarizer LLM
    )
    logging.info("Successfully loaded custom map and combine prompts for summarization chain.")
except Exception as e:
    logging.error(f"Failed to load map/combine prompts or create summarize chain: {e}")
    print(f"Error: Could not set up summarization chain with custom prompts.")
    exit(1)
# Setup for text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100) # Reduced chunk size for potentially faster summarization steps

def run_sustainability_report_analysis(industry: str) -> str:
    """
    Orchestrates the workflow to analyze sustainability reports for an industry.
    1. Identifies key companies.
    2. Searches for their latest sustainability reports (PDFs).
    3. Downloads and extracts text from found PDFs.
    4. Summarizes extracted texts.
    5. Synthesizes trends across summaries.
    """
    print(f"\n--- Starting Sustainability Report Analysis for: {industry} ---")
    all_extracted_texts = {} # Store extracted text: {company_name: text}
    report_urls_found = {} # Store found URLs: {company_name: url}

    # Step 1: Identify Key Companies
    print("\nStep 1: Identifying key companies...")
    try:
        # Load company identification prompt from file
        with open("internet_search_agent/company_identification_prompt.txt", "r") as f:
            company_prompt_template = f.read()
        company_prompt = company_prompt_template.format(industry=industry)
        company_response_message = llm.invoke(company_prompt)
        # Access the 'content' attribute of the AIMessage object
        company_response_content = company_response_message.content
        # Basic parsing, might need refinement
        companies = [name.strip() for name in company_response_content.split(',') if name.strip()]
        if not companies:
            return f"Error: Could not identify companies for the '{industry}' industry based on LLM response content: {company_response_content}"
        print(f"Identified companies: {', '.join(companies)}")
    except Exception as e:
        logging.error(f"Error identifying companies: {e}")
        return f"Error: Failed to identify companies using the LLM. {e}"

    # Step 2 & 3: Search, Download, Extract for each company
    print("\nStep 2 & 3: Searching for reports, downloading, and extracting text...")
    for company in companies:
        print(f"\nProcessing company: {company}")
        search_query = f"{company} sustainability report 2023 OR 2024 pdf filetype:pdf" # Target recent PDFs
        print(f"  Searching with query: '{search_query}'")
        search_results_str = search_langchain_tool.run(search_query)

        # Attempt to find a PDF URL in the search results
        pdf_url = None
        # Simple regex to find URLs ending in .pdf within the search results string
        pdf_links = re.findall(r'(https?://\S+\.pdf)', search_results_str, re.IGNORECASE)
        if pdf_links:
            pdf_url = pdf_links[0] # Take the first likely PDF link
            print(f"  Found potential PDF URL: {pdf_url}")
            report_urls_found[company] = pdf_url

            # Download
            print(f"  Downloading PDF...")
            local_path = download_pdf_tool.run(pdf_url)
            if local_path.startswith("Error"):
                print(f"  Download failed: {local_path}")
                continue # Skip to next company

            # Extract Text
            print(f"  Extracting text from: {local_path}")
            extracted_text = extract_pdf_text_tool.run(local_path)
            if extracted_text.startswith("Error") or extracted_text.startswith("Warning"):
                print(f"  Text extraction failed or yielded no text: {extracted_text}")
            else:
                print(f"  Successfully extracted text (length: {len(extracted_text)}).")
                all_extracted_texts[company] = extracted_text
        else:
            print(f"  Could not find a direct PDF link in search results for {company}.")

    if not all_extracted_texts:
        return "Analysis failed: No sustainability report text could be extracted for the identified companies."

    # Step 4: Summarize each document
    print("\nStep 4: Summarizing individual reports...")
    individual_summaries = {}
    for company, text in all_extracted_texts.items():
        print(f"  Summarizing report for {company}...")
        if not text.strip():
            print(f"  Skipping empty text for {company}.")
            individual_summaries[company] = "No text extracted."
            continue
        try:
            # Split the document into chunks
            docs = text_splitter.create_documents([text])
            # Run the map-reduce summarization chain
            summary = summarize_chain.run(docs)
            individual_summaries[company] = summary
            print(f"  Finished summarizing for {company}.")
        except Exception as e:
            logging.error(f"Error summarizing report for {company}: {e}")
            print(f"  Error summarizing report for {company}. Skipping.")
            individual_summaries[company] = f"Error during summarization: {e}"

    if not any(not s.startswith("Error") and s != "No text extracted." for s in individual_summaries.values()):
         return "Analysis failed: Could not generate summaries for any extracted reports."

    # Step 5: Synthesize trends from summaries
    print("\nStep 5: Synthesizing trends across summaries...")
    combined_summaries = "\n\n".join([f"--- Summary for {company} ---\n{summary}"
                                      for company, summary in individual_summaries.items()
                                      if not summary.startswith("Error") and summary != "No text extracted."])

    try:
        # Load synthesis prompt from file
        with open("internet_search_agent/synthesis_prompt.txt", "r") as f:
            synthesis_prompt_template = f.read()
        synthesis_prompt = synthesis_prompt_template.format(industry=industry, combined_summaries=combined_summaries)
        final_synthesis_message = llm.invoke(synthesis_prompt)
        # Access content from AIMessage
        final_synthesis = final_synthesis_message.content
        print("\n--- Analysis Complete ---")
        # Prepend the list of found report URLs for context
        report_list = "\n".join([f"- {comp}: {url}" for comp, url in report_urls_found.items()])
        if not report_list: report_list = "No report URLs were successfully identified or downloaded."

        return f"Analysis based on reports found for:\n{report_list}\n\n--- Synthesized Trends ---\n{final_synthesis}"
    except Exception as e:
        logging.error(f"Error during final synthesis: {e}")
        return f"Error: Failed during the final synthesis step. {e}"
