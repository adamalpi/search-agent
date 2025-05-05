import logging
import os
import re
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Updated import path again for InMemoryChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

# Import the initializer, not the instances directly
from research_agent.agent import initialize_gemini, load_prompt
from research_agent.file_tools import download_pdf_tool, extract_pdf_text_tool
from research_agent.history_tools import query_analysis_history_tool
from research_agent.research_tools import download_and_extract_reports, summarize_extracted_texts
from research_agent.search_tool import search_langchain_tool

logger = logging.getLogger(__name__)  # Add module-specific logger

# Define the list of tools for the basic agent node
BASIC_AGENT_TOOLS = [
    search_langchain_tool,
    download_pdf_tool,
    extract_pdf_text_tool,
    query_analysis_history_tool,
]

# logging.basicConfig(...) # Removed - Handled centrally

# --- Initialize LLM and Agent ---
# Initialize Gemini, capturing both the main LLM and the summarizer LLM
# Initialization moved into build_unified_graph function
# --- Define Graph State --- (Removed cache dir, text splitter, summarize chain setup)


# --- Define Unified Graph State ---


# We use MessagesState for chat_history, which LangGraph handles efficiently.
# We add Optional[] to fields that might not be present in every path.
class UnifiedGraphState(MessagesState):  # Inherit from MessagesState for chat_history
    """Represents the state of our unified graph."""

    # Inputs (mutually exclusive for routing)
    input_query: Optional[str] = None
    industry: Optional[str] = None

    # Basic Agent fields
    agent_response: Optional[str] = None
    # chat_history is inherited from MessagesState (key: "messages")

    # Research Agent fields
    companies: Optional[List[str]] = None
    report_urls: Optional[Dict[str, str]] = None  # {company: url}
    extracted_texts: Optional[Dict[str, str]] = None  # {company: text}
    individual_summaries: Optional[Dict[str, str]] = None  # {company: summary}
    synthesis_result: Optional[str] = None
    error_message: Optional[str] = None  # To capture errors during execution


# --- Define Graph Nodes ---


# Removed module-level agent initialization. It will happen inside build_unified_graph.

# --- Define Graph Nodes (These will be defined *inside* build_unified_graph) ---


# --- Router Node (Can stay outside as it doesn't need LLMs/chains) ---
# Note: This node's primary function is routing via conditional edges.
# It should return an empty dict as it doesn't modify the state directly.
# The conditional edge logic will call this function to get the routing string.
def route_request(state: UnifiedGraphState) -> dict:
    """Determines the route and returns an empty state update."""
    print("--- Node: route_request ---")
    if state.get("industry"):
        print("Decision: Route to research path.")
        # The string "research_path" is implicitly used by the conditional edge logic
    elif state.get("input_query"):
        print("Decision: Route to basic agent path.")
        # The string "basic_agent_path" is implicitly used by the conditional edge logic
    else:
        print("Decision: Route to error: No valid input found.")
        # The string "error" is implicitly used by the conditional edge logic
    return {}  # Return empty dict as this node doesn't update state


# --- Conditional Edge Logic Function ---
# This function is explicitly used by add_conditional_edges to get the routing key.
def decide_route(state: UnifiedGraphState) -> str:
    """Returns the routing key based on the input state."""
    if state.get("industry"):
        return "research_path"
    elif state.get("input_query"):
        return "basic_agent_path"
    else:
        return "error"


# --- Basic Agent Node ---
# --- Research Path Nodes (Modified State Type) ---


def search_for_reports(state: UnifiedGraphState) -> UnifiedGraphState:
    """Searches for sustainability reports for each identified company."""
    print("--- Node: search_for_reports ---")
    if state.get("error_message"):
        return {}  # Stop if previous node failed
    companies = state["companies"]
    report_urls = state.get("report_urls", {})
    logger.info(f"Searching for reports for companies: {companies}")  # Use logger

    for company in companies:
        if company in report_urls:
            continue  # Skip if already processed (e.g., retry)
        logger.info(f"Searching for report for: {company}")  # Use logger
        search_query = f"{company} sustainability report 2023 OR 2024 pdf filetype:pdf"
        try:
            search_results_str = search_langchain_tool.run(search_query)
            # Find the first PDF link
            pdf_links = re.findall(r"(https?://\S+\.pdf)", search_results_str, re.IGNORECASE)
            if pdf_links:
                pdf_url = pdf_links[0]
                logger.info(f"Found potential PDF URL for {company}: {pdf_url}")  # Use logger
                report_urls[company] = pdf_url
            else:
                logger.warning(f"No direct PDF link found in search results for {company}.")  # Use logger
                # Store None or skip? Storing None for now to indicate search was attempted.
                report_urls[company] = None
        except Exception as e:
            logger.error(f"Error searching for report for {company}: {e}", exc_info=True)  # Use logger
            report_urls[company] = f"Error during search: {e}"  # Store error indication

    return {"report_urls": report_urls}


def download_and_extract(state: UnifiedGraphState) -> UnifiedGraphState:
    """Node wrapper for downloading PDFs and extracting text using research_tools."""
    print("--- Node: download_and_extract (using research_tools) ---")
    if state.get("error_message"):
        logger.warning("Skipping download_and_extract due to previous error.")  # Use logger
        return {}
    report_urls = state.get("report_urls")
    if not report_urls:
        logger.warning("No report URLs found in state for download_and_extract.")  # Use logger
        # Decide if this is an error or just an empty step
        return {"extracted_texts": {}}  # Return empty dict if no URLs

    try:
        extracted_texts = download_and_extract_reports(report_urls)
        return {"extracted_texts": extracted_texts}
    except Exception as e:
        logger.error(f"Error calling download_and_extract_reports: {e}", exc_info=True)  # Use logger
        # Populate errors for all expected companies? Or just return a general error?
        error_texts = {company: f"Error in download/extract step: {e}" for company in report_urls}
        return {
            "extracted_texts": error_texts,
            "error_message": f"Failed during download/extraction: {e}",
        }


def summarize_reports(state: UnifiedGraphState) -> UnifiedGraphState:
    """Node wrapper for summarizing extracted text using research_tools."""
    print("--- Node: summarize_reports (using research_tools) ---")
    if state.get("error_message"):
        logger.warning("Skipping summarize_reports due to previous error.")  # Use logger
        return {}
    extracted_texts = state.get("extracted_texts")
    if not extracted_texts:
        logger.warning("No extracted texts found in state for summarize_reports.")  # Use logger
        return {"individual_summaries": {}}  # Return empty dict if no texts

    try:
        individual_summaries = summarize_extracted_texts(extracted_texts)
        return {"individual_summaries": individual_summaries}
    except Exception as e:
        logger.error(f"Error calling summarize_extracted_texts: {e}", exc_info=True)  # Use logger
        error_summaries = {company: f"Error in summarization step: {e}" for company in extracted_texts}
        return {
            "individual_summaries": error_summaries,
            "error_message": f"Failed during summarization: {e}",
        }


# --- Error Handling Node ---
def handle_error(state: UnifiedGraphState) -> UnifiedGraphState:
    """Handles errors captured in the state."""
    print("--- Node: handle_error ---")
    error = state.get("error_message", "Unknown error")
    logger.error(f"Graph execution failed: {error}")  # Use logger
    # We can potentially add more sophisticated error handling/reporting here
    # For now, just ensure the error message is in the final state
    return {"error_message": error}


# --- Build the Unified Graph ---


def build_unified_graph():
    """Builds the unified LangGraph workflow, initializing components inside."""

    # --- Initialize LLMs and Chains ---
    try:
        llm, llm_summarizer = initialize_gemini()
        react_prompt_template = load_prompt()

        # Load prompts for summarization chain
        prompts_dir = "prompts"  # Relative to execution root
        with open(os.path.join(prompts_dir, "map_prompt.txt"), "r") as f:
            map_prompt_template = f.read()
        map_prompt = PromptTemplate.from_template(map_prompt_template)
        with open(os.path.join(prompts_dir, "combine_prompt.txt"), "r") as f:
            combine_prompt_template = f.read()
        combine_prompt = PromptTemplate.from_template(combine_prompt_template)

        # Initialize summarize chain
        summarize_chain = load_summarize_chain(
            llm=llm_summarizer,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )
        logger.info("Graph Builder: Summarize chain initialized.")  # Use logger

        # Initialize ReAct agent (without executor)
        react_agent = create_react_agent(llm, BASIC_AGENT_TOOLS, react_prompt_template)
        logger.info("Graph Builder: ReAct agent initialized.")  # Use logger

    except Exception as e:
        logger.error(f"Graph Builder: Failed to initialize LLMs or chains: {e}", exc_info=True)  # Use logger
        # Handle initialization failure - maybe raise exception or return a dummy graph?
        raise RuntimeError(f"Failed to initialize graph components: {e}") from e

    # --- Define Node Functions (Nested within build_unified_graph) ---
    # These functions now have access to llm, llm_summarizer, summarize_chain, react_agent

    def run_basic_agent(state: UnifiedGraphState) -> UnifiedGraphState:
        """Executes the basic ReAct agent for general queries."""
        print("--- Node: run_basic_agent ---")
        query = state["input_query"]
        messages = state.get("messages", [])
        temp_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=InMemoryChatMessageHistory(),
        )
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_memory.chat_memory.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                temp_memory.chat_memory.add_ai_message(msg.content)

        # Use the react_agent initialized in the outer scope
        react_agent_executor = AgentExecutor(
            agent=react_agent,
            tools=BASIC_AGENT_TOOLS,
            memory=temp_memory,
            verbose=True,
            handle_parsing_errors=True,
        )
        logger.info(f"Running basic agent with query: {query}")  # Use logger
        try:
            response = react_agent_executor.invoke({"input": query})
            agent_response = response.get("output", "Agent did not provide a final answer.")
            logger.info(f"Basic agent response: {agent_response}")  # Use logger
            updated_messages = temp_memory.chat_memory.messages
            return {"messages": updated_messages, "agent_response": agent_response}
        except Exception as e:
            logger.error(f"Error in run_basic_agent: {e}", exc_info=True)  # Use logger
            return {"error_message": f"Failed during basic agent execution: {e}"}

    def identify_companies(state: UnifiedGraphState) -> UnifiedGraphState:
        """Identifies key companies for the given industry."""
        print("--- Node: identify_companies ---")
        if state.get("error_message"):
            return {}
        industry = state["industry"]
        logger.info(f"Identifying companies for industry: {industry}")  # Use logger
        try:
            with open("prompts/company_identification_prompt.txt", "r") as f:  # Path relative to execution root
                company_prompt_template = f.read()
            company_prompt = company_prompt_template.format(industry=industry)
            # Use llm from outer scope
            company_response_message = llm.invoke(company_prompt)
            company_response_content = company_response_message.content
            companies = [name.strip() for name in company_response_content.split(",") if name.strip()]
            if not companies:
                logger.warning(  # Use logger
                    f"LLM did not identify companies for industry '{industry}'. Response: {company_response_content}"
                )
                return {"error_message": f"Could not identify companies for industry '{industry}'."}
            logger.info(f"Identified companies: {companies}")  # Use logger
            return {
                "companies": companies,
                "report_urls": {},
                "extracted_texts": {},
                "individual_summaries": {},
            }
        except Exception as e:
            logger.error(f"Error in identify_companies: {e}", exc_info=True)  # Use logger
            return {"error_message": f"Failed to identify companies: {e}"}

    def search_for_reports(state: UnifiedGraphState) -> UnifiedGraphState:
        """Searches for sustainability reports for each identified company."""
        print("--- Node: search_for_reports ---")
        if state.get("error_message"):
            return {}
        companies = state["companies"]
        report_urls = state.get("report_urls", {})
        logger.info(f"Searching for reports for companies: {companies}")  # Use logger
        for company in companies:
            if company in report_urls:
                continue
            logger.info(f"Searching for report for: {company}")  # Use logger
            search_query = f"{company} sustainability report 2023 OR 2024 pdf filetype:pdf"
            try:
                search_results_str = search_langchain_tool.run(search_query)
                pdf_links = re.findall(r"(https?://\S+\.pdf)", search_results_str, re.IGNORECASE)
                if pdf_links:
                    pdf_url = pdf_links[0]
                    logger.info(f"Found potential PDF URL for {company}: {pdf_url}")  # Use logger
                    report_urls[company] = pdf_url
                else:
                    logger.warning(f"No direct PDF link found in search results for {company}.")  # Use logger
                    report_urls[company] = None
            except Exception as e:
                logger.error(f"Error searching for report for {company}: {e}", exc_info=True)  # Use logger
                report_urls[company] = f"Error during search: {e}"
        return {"report_urls": report_urls}

    def download_and_extract(state: UnifiedGraphState) -> UnifiedGraphState:
        """Node wrapper for downloading PDFs and extracting text using research_tools."""
        print("--- Node: download_and_extract (using research_tools) ---")
        if state.get("error_message"):
            logger.warning("Skipping download_and_extract due to previous error.")  # Use logger
            return {}
        report_urls = state.get("report_urls")
        if not report_urls:
            logger.warning("No report URLs found in state for download_and_extract.")  # Use logger
            return {"extracted_texts": {}}
        try:
            # Call the function from research_tools
            extracted_texts = download_and_extract_reports(report_urls)
            return {"extracted_texts": extracted_texts}
        except Exception as e:
            logger.error(f"Error calling download_and_extract_reports: {e}", exc_info=True)  # Use logger
            error_texts = {company: f"Error in download/extract step: {e}" for company in report_urls}
            return {
                "extracted_texts": error_texts,
                "error_message": f"Failed during download/extraction: {e}",
            }

    def summarize_reports(state: UnifiedGraphState) -> UnifiedGraphState:
        """Node wrapper for summarizing extracted text using research_tools."""
        print("--- Node: summarize_reports (using research_tools) ---")
        if state.get("error_message"):
            logger.warning("Skipping summarize_reports due to previous error.")  # Use logger
            return {}
        extracted_texts = state.get("extracted_texts")
        if not extracted_texts:
            logger.warning("No extracted texts found in state for summarize_reports.")  # Use logger
            return {"individual_summaries": {}}
        try:
            # Call the function from research_tools, passing the summarize_chain from outer scope
            individual_summaries = summarize_extracted_texts(extracted_texts, summarize_chain)
            return {"individual_summaries": individual_summaries}
        except Exception as e:
            logger.error(f"Error calling summarize_extracted_texts: {e}", exc_info=True)  # Use logger
            error_summaries = {company: f"Error in summarization step: {e}" for company in extracted_texts}
            return {
                "individual_summaries": error_summaries,
                "error_message": f"Failed during summarization: {e}",
            }

    def synthesize_trends(state: UnifiedGraphState) -> UnifiedGraphState:
        """Synthesizes trends from the individual summaries."""
        print("--- Node: synthesize_trends ---")
        if state.get("error_message"):
            return {}
        industry = state["industry"]
        individual_summaries = state["individual_summaries"]
        report_urls = state["report_urls"]
        valid_summaries = {
            comp: summary
            for comp, summary in individual_summaries.items()
            if summary and not summary.startswith("Error") and not summary.startswith("Skipped")
        }
        if not valid_summaries:
            logger.warning("No valid summaries available for synthesis.")  # Use logger
            return {"error_message": "Analysis failed: No valid summaries could be generated."}
        logger.info("Synthesizing trends across summaries.")  # Use logger
        combined_summaries = "\n\n".join(
            [f"--- Summary for {company} ---\n{summary}" for company, summary in valid_summaries.items()]
        )
        try:
            with open("prompts/synthesis_prompt.txt", "r") as f:  # Path relative to execution root
                synthesis_prompt_template = f.read()
            synthesis_prompt = synthesis_prompt_template.format(
                industry=industry, combined_summaries=combined_summaries
            )
            # Use llm from outer scope
            final_synthesis_message = llm.invoke(synthesis_prompt)
            final_synthesis = final_synthesis_message.content
            report_list = "\n".join(
                [f"- {comp}: {report_urls.get(comp, 'URL not found/processed')}" for comp in valid_summaries.keys()]
            )
            if not report_list:
                report_list = "No report URLs were successfully processed."
            # Construct the final result string step-by-step for readability and line length
            result_header = "Analysis based on reports processed for:\n"
            report_section = f"{report_list}\n\n"
            trends_header = "--- Synthesized Trends ---\n"
            final_result = result_header + report_section + trends_header + final_synthesis
            logger.info("Finished synthesizing trends.")  # Use logger
            return {"synthesis_result": final_result}
        except Exception as e:
            logger.error(f"Error during final synthesis: {e}", exc_info=True)  # Use logger
            return {"error_message": f"Failed during the final synthesis step: {e}"}

    def handle_error(state: UnifiedGraphState) -> UnifiedGraphState:
        """Handles errors captured in the state."""
        print("--- Node: handle_error ---")
        error = state.get("error_message", "Unknown error")
        logger.error(f"Graph execution failed: {error}")  # Use logger
        return {"error_message": error}

    def route_research_step(state: UnifiedGraphState) -> str:
        """Checks for errors and decides whether to continue or handle error."""
        if state.get("error_message"):
            logger.warning("Error detected in research step, routing to handle_error.")  # Use logger
            return "handle_error"
        else:
            logger.info("No error detected in research step, continuing.")  # Use logger
            return "continue"

    # --- Build the Graph Structure ---
    workflow = StateGraph(UnifiedGraphState)

    # Add nodes using the nested functions
    workflow.add_node("route_request", route_request)  # Router can stay outside
    workflow.add_node("run_basic_agent", run_basic_agent)
    workflow.add_node("identify_companies", identify_companies)
    workflow.add_node("search_for_reports", search_for_reports)
    workflow.add_node("download_and_extract", download_and_extract)
    workflow.add_node("summarize_reports", summarize_reports)
    workflow.add_node("synthesize_trends", synthesize_trends)
    workflow.add_node("handle_error", handle_error)

    # Set entry point
    workflow.set_entry_point("route_request")

    # Define conditional edges from the router
    # Define conditional edges from the router node
    # Use the dedicated 'decide_route' function for the routing logic
    workflow.add_conditional_edges(
        "route_request",
        decide_route,
        {
            "basic_agent_path": "run_basic_agent",
            "research_path": "identify_companies",
            "error": "handle_error",
        },
    )

    # Define conditional edges for the research path
    workflow.add_conditional_edges(
        "identify_companies",
        route_research_step,
        {"continue": "search_for_reports", "handle_error": "handle_error"},
    )
    workflow.add_conditional_edges(
        "search_for_reports",
        route_research_step,
        {"continue": "download_and_extract", "handle_error": "handle_error"},
    )
    workflow.add_conditional_edges(
        "download_and_extract",
        route_research_step,
        {"continue": "summarize_reports", "handle_error": "handle_error"},
    )
    workflow.add_conditional_edges(
        "summarize_reports",
        route_research_step,
        {"continue": "synthesize_trends", "handle_error": "handle_error"},
    )
    # Final step still needs routing based on its own potential errors or success
    workflow.add_conditional_edges(
        "synthesize_trends",
        route_research_step,  # Use the same checker, map 'continue' to END
        {"continue": END, "handle_error": "handle_error"},
    )

    # Define end points for other paths
    workflow.add_edge("run_basic_agent", END)
    workflow.add_edge("handle_error", END)  # End after handling error

    # Compile the workflow
    app = workflow.compile()
    logging.info("Unified LangGraph workflow compiled.")
    return app
