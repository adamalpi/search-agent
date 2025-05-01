import logging
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
# Import necessary components initialized in agent.py (or initialize them here if preferred)
# For now, assume they are passed in or globally accessible for simplicity,
# but dependency injection is better practice for larger apps.
# Use absolute imports assuming 'internet_search_agent' is in the Python path
from agent import llm, llm_summarizer # Import only LLMs from agent.py
from search_tool import search_langchain_tool
from file_tools import download_pdf_tool, extract_pdf_text_tool
from langchain.prompts import PromptTemplate # Added missing import
from langchain.chains.summarize import load_summarize_chain # Added missing import
from langchain_text_splitters import RecursiveCharacterTextSplitter # Added missing import
import re
import os # Import os for file path operations

# Configure logging (can inherit from agent.py if run together)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define Cache Directory ---
SUMMARY_CACHE_DIR = os.path.join(os.path.dirname(__file__), "summary_cache")
os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)
logging.info(f"Using summary cache directory: {SUMMARY_CACHE_DIR}")

# --- Initialize Components Used by the Graph ---

# Setup for text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
logging.info("Initialized text splitter for graph.")

# Load map and combine prompts for summarization chain
try:
    with open("internet_search_agent/prompts/map_prompt.txt", "r") as f: # Updated path
        map_prompt_template = f.read()
    map_prompt = PromptTemplate.from_template(map_prompt_template)

    with open("internet_search_agent/prompts/combine_prompt.txt", "r") as f: # Updated path
        combine_prompt_template = f.read()
    combine_prompt = PromptTemplate.from_template(combine_prompt_template)

    # Setup for summarization chain (using the dedicated summarizer LLM)
    summarize_chain = load_summarize_chain(
        llm=llm_summarizer, # Imported from agent.py
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True
    )
    logging.info("Successfully loaded custom map/combine prompts and created summarize chain for graph.")
except Exception as e:
    logging.error(f"Failed to load map/combine prompts or create summarize chain in graph_builder: {e}", exc_info=True)
    print(f"FATAL: Could not set up summarization chain. Exiting.")
    exit(1)


# --- Define Graph State ---

class GraphState(TypedDict):
    """Represents the state of our graph."""
    industry: str
    companies: List[str] = None
    report_urls: Dict[str, str] = None # {company: url}
    extracted_texts: Dict[str, str] = None # {company: text}
    individual_summaries: Dict[str, str] = None # {company: summary}
    synthesis_result: str = None
    error_message: str = None # To capture errors during execution

# --- Define Graph Nodes ---

def identify_companies(state: GraphState) -> GraphState:
    """Identifies key companies for the given industry."""
    print("--- Node: identify_companies ---")
    industry = state['industry']
    logging.info(f"Identifying companies for industry: {industry}")
    try:
        # Load company identification prompt from file
        with open("internet_search_agent/prompts/company_identification_prompt.txt", "r") as f: # Updated path
            company_prompt_template = f.read()
        company_prompt = company_prompt_template.format(industry=industry)
        company_response_message = llm.invoke(company_prompt)
        company_response_content = company_response_message.content
        companies = [name.strip() for name in company_response_content.split(',') if name.strip()]

        if not companies:
            logging.warning(f"LLM did not identify companies for industry '{industry}'. Response: {company_response_content}")
            return {"error_message": f"Could not identify companies for industry '{industry}'."}

        logging.info(f"Identified companies: {companies}")
        return {"companies": companies, "report_urls": {}, "extracted_texts": {}, "individual_summaries": {}} # Initialize downstream dicts
    except Exception as e:
        logging.error(f"Error in identify_companies: {e}", exc_info=True)
        return {"error_message": f"Failed to identify companies: {e}"}

def search_for_reports(state: GraphState) -> GraphState:
    """Searches for sustainability reports for each identified company."""
    print("--- Node: search_for_reports ---")
    if state.get("error_message"): return {} # Stop if previous node failed
    companies = state['companies']
    report_urls = state.get('report_urls', {})
    logging.info(f"Searching for reports for companies: {companies}")

    for company in companies:
        if company in report_urls: continue # Skip if already processed (e.g., retry)
        logging.info(f"Searching for report for: {company}")
        search_query = f"{company} sustainability report 2023 OR 2024 pdf filetype:pdf"
        try:
            search_results_str = search_langchain_tool.run(search_query)
            # Find the first PDF link
            pdf_links = re.findall(r'(https?://\S+\.pdf)', search_results_str, re.IGNORECASE)
            if pdf_links:
                pdf_url = pdf_links[0]
                logging.info(f"Found potential PDF URL for {company}: {pdf_url}")
                report_urls[company] = pdf_url
            else:
                logging.warning(f"No direct PDF link found in search results for {company}.")
                # Store None or skip? Storing None for now to indicate search was attempted.
                report_urls[company] = None
        except Exception as e:
            logging.error(f"Error searching for report for {company}: {e}", exc_info=True)
            report_urls[company] = f"Error during search: {e}" # Store error indication

    return {"report_urls": report_urls}

def download_and_extract(state: GraphState) -> GraphState:
    """Downloads PDFs and extracts text."""
    print("--- Node: download_and_extract ---")
    if state.get("error_message"): return {}
    report_urls = state['report_urls']
    extracted_texts = state.get('extracted_texts', {})
    logging.info("Downloading and extracting text from found PDFs.")

    for company, url in report_urls.items():
        if company in extracted_texts: continue # Skip if already processed
        if url is None or url.startswith("Error"):
            logging.warning(f"Skipping download/extraction for {company} due to missing/error URL: {url}")
            extracted_texts[company] = "No valid URL found."
            continue

        logging.info(f"Processing PDF for {company} from URL: {url}")
        try:
            # Download (uses cache)
            local_path = download_pdf_tool.run(url)
            if local_path.startswith("Error"):
                logging.error(f"Download failed for {company}: {local_path}")
                extracted_texts[company] = f"Download failed: {local_path}"
                continue

            # Extract Text
            extracted_text = extract_pdf_text_tool.run(local_path)
            if extracted_text.startswith("Error") or extracted_text.startswith("Warning"):
                logging.warning(f"Text extraction failed/empty for {company}: {extracted_text}")
                extracted_texts[company] = extracted_text # Store warning/error
            else:
                logging.info(f"Successfully extracted text for {company} (length: {len(extracted_text)}).")
                extracted_texts[company] = extracted_text
        except Exception as e:
            logging.error(f"Error during download/extraction for {company}: {e}", exc_info=True)
            extracted_texts[company] = f"Error during download/extraction: {e}"

    return {"extracted_texts": extracted_texts}

def summarize_reports(state: GraphState) -> GraphState:
    """Summarizes the extracted text for each report, using a cache."""
    print("--- Node: summarize_reports (with caching) ---")
    if state.get("error_message"): return {}
    extracted_texts = state['extracted_texts']
    individual_summaries = state.get('individual_summaries', {})
    logging.info("Summarizing extracted report texts.")

    for company, text in extracted_texts.items():
        if company in individual_summaries: continue # Skip if already processed
        if text is None or text.startswith("Error") or text.startswith("Warning") or text == "No valid URL found." or not text.strip():
            logging.warning(f"Skipping summarization for {company} due to previous error or empty text.")
            individual_summaries[company] = "Skipped due to previous error or empty text."
            continue

        # Check cache first
        # Use a simple filename based on company name for now.
        cache_filename = f"{company.replace(' ', '_').lower()}_summary.txt"
        cache_filepath = os.path.join(SUMMARY_CACHE_DIR, cache_filename)

        if os.path.exists(cache_filepath):
            logging.info(f"Cache hit for {company} summary: Loading from {cache_filepath}")
            try:
                with open(cache_filepath, "r") as f:
                    summary = f.read()
                individual_summaries[company] = summary
                continue # Move to next company
            except Exception as e:
                logging.warning(f"Failed to read cached summary for {company} from {cache_filepath}: {e}. Will regenerate.")

        # If cache miss or read error, generate summary
        logging.info(f"Cache miss for {company}. Generating summary...")
        try:
            docs = text_splitter.create_documents([text])
            summary = summarize_chain.run(docs) # Uses llm_summarizer with timeout
            individual_summaries[company] = summary
            logging.info(f"Finished generating summary for {company}.")

            # Save summary to cache
            try:
                with open(cache_filepath, "w") as f:
                    f.write(summary)
                logging.info(f"Saved summary for {company} to cache: {cache_filepath}")
            except Exception as e:
                logging.error(f"Failed to save summary to cache for {company} at {cache_filepath}: {e}")

        except Exception as e:
            logging.error(f"Error summarizing report for {company}: {e}", exc_info=True)
            individual_summaries[company] = f"Error during summarization: {e}"

    return {"individual_summaries": individual_summaries}

def synthesize_trends(state: GraphState) -> GraphState:
    """Synthesizes trends from the individual summaries."""
    print("--- Node: synthesize_trends ---")
    if state.get("error_message"): return {}
    industry = state['industry']
    individual_summaries = state['individual_summaries']
    report_urls = state['report_urls'] # Get URLs to include in final output

    valid_summaries = {comp: summary for comp, summary in individual_summaries.items()
                       if summary and not summary.startswith("Error") and not summary.startswith("Skipped")}

    if not valid_summaries:
        logging.warning("No valid summaries available for synthesis.")
        return {"error_message": "Analysis failed: No valid summaries could be generated."}

    logging.info("Synthesizing trends across summaries.")
    combined_summaries = "\n\n".join([f"--- Summary for {company} ---\n{summary}"
                                      for company, summary in valid_summaries.items()])

    try:
        # Load synthesis prompt from file
        with open("internet_search_agent/prompts/synthesis_prompt.txt", "r") as f: # Updated path
            synthesis_prompt_template = f.read()
        synthesis_prompt = synthesis_prompt_template.format(industry=industry, combined_summaries=combined_summaries)
        final_synthesis_message = llm.invoke(synthesis_prompt) # Use main LLM
        final_synthesis = final_synthesis_message.content

        # Format final result with list of sources
        report_list = "\n".join([f"- {comp}: {report_urls.get(comp, 'URL not found/processed')}" for comp in valid_summaries.keys()])
        if not report_list: report_list = "No report URLs were successfully processed."

        final_result = f"Analysis based on reports processed for:\n{report_list}\n\n--- Synthesized Trends ---\n{final_synthesis}"
        logging.info("Finished synthesizing trends.")
        return {"synthesis_result": final_result}
    except Exception as e:
        logging.error(f"Error during final synthesis: {e}", exc_info=True)
        return {"error_message": f"Failed during the final synthesis step: {e}"}

# --- Build the Graph ---

def build_graph():
    """Builds the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("identify_companies", identify_companies)
    workflow.add_node("search_for_reports", search_for_reports)
    workflow.add_node("download_and_extract", download_and_extract)
    workflow.add_node("summarize_reports", summarize_reports)
    workflow.add_node("synthesize_trends", synthesize_trends)

    # Build graph: Set entry point and define edges
    workflow.set_entry_point("identify_companies")
    workflow.add_edge("identify_companies", "search_for_reports")
    workflow.add_edge("search_for_reports", "download_and_extract")
    workflow.add_edge("download_and_extract", "summarize_reports")
    workflow.add_edge("summarize_reports", "synthesize_trends")
    workflow.add_edge("synthesize_trends", END) # End after synthesis

    # Compile the workflow
    app = workflow.compile()
    logging.info("LangGraph workflow compiled.")
    return app

# --- Main execution (for testing graph directly) ---
if __name__ == '__main__':
    print("Building and testing graph...")
    graph_app = build_graph()

    # Example invocation
    test_industry = "automotive"
    print(f"\nInvoking graph for industry: {test_industry}")
    inputs = {"industry": test_industry}
    try:
        # Stream events to see node execution
        for event in graph_app.stream(inputs):
            print(event)
            # The final state is typically in the last event or accessible differently
            # depending on how you want to retrieve it.
            # For simplicity, we'll just print events here.

        # To get the final state after streaming:
        final_state = graph_app.invoke(inputs)
        print("\n--- Final State ---")
        print(final_state)
        print("\n--- Synthesis Result ---")
        print(final_state.get("synthesis_result") or final_state.get("error_message", "Unknown error"))

    except Exception as e:
        print(f"\nError invoking graph: {e}")