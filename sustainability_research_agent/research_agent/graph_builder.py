import functools
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from research_agent.agent import initialize_gemini, load_prompt
from research_agent.file_tools import download_pdf_tool, extract_pdf_text_tool
from research_agent.history_tools import query_analysis_history_tool
from research_agent.research_tools import download_and_extract_reports, summarize_extracted_texts
from research_agent.search_tool import search_langchain_tool

logger = logging.getLogger(__name__)

BASIC_AGENT_TOOLS = [
    search_langchain_tool,
    download_pdf_tool,
    extract_pdf_text_tool,
    query_analysis_history_tool,
]

# --- Configuration Constants ---
CURRENT_FILE_DIR = Path(__file__).parent
PROMPTS_DIR = CURRENT_FILE_DIR.parent / "prompts"

COMPANY_ID_PROMPT_FILE = PROMPTS_DIR / "company_identification_prompt.txt"
SYNTHESIS_PROMPT_FILE = PROMPTS_DIR / "synthesis_prompt.txt"
MAP_PROMPT_FILE = PROMPTS_DIR / "map_prompt.txt"
COMBINE_PROMPT_FILE = PROMPTS_DIR / "combine_prompt.txt"

MAX_SEARCH_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# --- Routing Constants ---
ROUTE_RESEARCH_PATH = "research_path"
ROUTE_BASIC_AGENT_PATH = "basic_agent_path"
ROUTE_ERROR_INPUT = "error"
ROUTE_CONTINUE = "continue"
ROUTE_HANDLE_ERROR_STEP = "handle_error"


class UnifiedGraphState(MessagesState):
    """Represents the state of our unified graph."""

    input_query: Optional[str] = None
    industry: Optional[str] = None

    agent_response: Optional[str] = None

    companies: Optional[List[str]] = None
    report_urls: Optional[Dict[str, str]] = None
    extracted_texts: Optional[Dict[str, str]] = None
    individual_summaries: Optional[Dict[str, str]] = None
    synthesis_result: Optional[str] = None
    error_message: Optional[str] = None


def route_request(state: UnifiedGraphState) -> dict:
    """Determines the route and returns an empty state update."""
    print("--- Node: route_request ---")
    if state.get("industry"):
        print("Decision: Route to research path.")
    elif state.get("input_query"):
        print("Decision: Route to basic agent path.")
    else:
        print("Decision: Route to error: No valid input found.")
    return {}


def decide_route(state: UnifiedGraphState) -> str:
    """Returns the routing key based on the input state."""
    if state.get("industry"):
        return ROUTE_RESEARCH_PATH
    elif state.get("input_query"):
        return ROUTE_BASIC_AGENT_PATH
    else:
        return ROUTE_ERROR_INPUT


# --- Extracted Node Functions ---
def _identify_companies_node(state: UnifiedGraphState, llm, company_prompt_template_str: str) -> UnifiedGraphState:
    """Identifies key companies for the given industry."""
    print("--- Node: identify_companies ---")
    if state.get("error_message"):
        return {}
    industry = state["industry"]
    logger.info(f"Identifying companies for industry: {industry}")
    try:
        company_prompt = company_prompt_template_str.format(industry=industry)
        company_response_message = llm.invoke(company_prompt)
        company_response_content = company_response_message.content
        companies = [name.strip() for name in company_response_content.split(",") if name.strip()]
        if not companies:
            logger.warning(
                f"LLM did not identify companies for industry '{industry}'. Response: {company_response_content}"
            )
            return {"error_message": f"Could not identify companies for industry '{industry}'."}
        logger.info(f"Identified companies: {companies}")
        return {
            "companies": companies,
            "report_urls": {},
            "extracted_texts": {},
            "individual_summaries": {},
        }
    except Exception as e:
        logger.error(f"Error in identify_companies: {e}", exc_info=True)
        return {"error_message": f"Failed to identify companies: {e}"}


def _run_basic_agent_node(state: UnifiedGraphState, react_agent, basic_agent_tools) -> UnifiedGraphState:
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

    react_agent_executor = AgentExecutor(
        agent=react_agent,
        tools=basic_agent_tools,
        memory=temp_memory,
        verbose=False,
        handle_parsing_errors=True,
    )
    logger.info(f"Running basic agent with query: {query}")
    try:
        response = react_agent_executor.invoke({"input": query})
        agent_response = response.get("output", "Agent did not provide a final answer.")
        logger.info(f"Basic agent response: {agent_response}")
        updated_messages = temp_memory.chat_memory.messages
        return {"messages": updated_messages, "agent_response": agent_response}
    except Exception as e:
        logger.error(f"Error in run_basic_agent: {e}", exc_info=True)
        return {"error_message": f"Failed during basic agent execution: {e}"}


def _search_for_reports_node(state: UnifiedGraphState) -> UnifiedGraphState:
    """Searches for sustainability reports for each identified company with retries."""
    print("--- Node: search_for_reports ---")
    if state.get("error_message"):
        return {}
    companies = state["companies"]
    report_urls = state.get("report_urls", {})
    logger.info(f"Searching for reports for companies: {companies}")

    for company in companies:
        if company in report_urls and (
            report_urls[company] is None or (report_urls[company] and not str(report_urls[company]).startswith("Error"))
        ):
            logger.info(f"Report URL or null marker already present for {company}, skipping search.")
            continue

        logger.info(f"Searching for report for: {company}")
        search_query = f"{company} sustainability report filetype:pdf"
        logger.info(f"Using search query: {search_query}")

        for attempt in range(MAX_SEARCH_RETRIES):
            try:
                search_results_str = search_langchain_tool.run(search_query)
                if "Error during DuckDuckGo search" in search_results_str or "Ratelimit" in search_results_str:
                    raise Exception(f"Search tool returned an error: {search_results_str}")

                pdf_links = re.findall(r"(https?://\S+\.pdf)", search_results_str, re.IGNORECASE)
                if pdf_links:
                    pdf_url = pdf_links[0]
                    logger.info(f"Found potential PDF URL for {company}: {pdf_url}")
                    report_urls[company] = pdf_url
                    break
                else:
                    if attempt == MAX_SEARCH_RETRIES - 1:
                        logger.warning(f"No direct PDF link found for {company} after {MAX_SEARCH_RETRIES} attempts.")
                        report_urls[company] = None
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1}/{MAX_SEARCH_RETRIES} for {company} failed: {e}")
                if attempt < MAX_SEARCH_RETRIES - 1:
                    logger.info(f"Retrying search for {company} in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.error(f"All {MAX_SEARCH_RETRIES} search attempts failed for {company}: {e}", exc_info=False)
                    report_urls[company] = f"Error during search for {company} after retries: {e}"
                    break

        if company not in report_urls:
            report_urls[company] = None

    return {"report_urls": report_urls}


def _download_and_extract_node(state: UnifiedGraphState) -> UnifiedGraphState:
    """Node wrapper for downloading PDFs and extracting text using research_tools."""
    print("--- Node: download_and_extract (using research_tools) ---")
    if state.get("error_message"):
        logger.warning("Skipping download_and_extract due to previous error.")
        return {}
    report_urls = state.get("report_urls")
    if not report_urls:
        logger.warning("No report URLs found in state for download_and_extract.")
        return {"extracted_texts": {}}
    try:
        extracted_texts_results = download_and_extract_reports(report_urls)

        for company, text_or_error in extracted_texts_results.items():
            if isinstance(text_or_error, str) and (
                text_or_error.startswith("Error:") or text_or_error.startswith("Warning:")
            ):
                logger.warning(f"Download/extraction for {company} resulted in: {text_or_error}")

        logger.info("Download and extraction process completed for all companies (some may have failed individually).")
        return {"extracted_texts": extracted_texts_results}

    except Exception as e:
        logger.error(f"Critical unexpected error in download_and_extract node orchestration: {e}", exc_info=True)
        error_texts = {
            company: f"Critical error in download/extract node orchestration: {e}" for company in report_urls
        }
        return {"extracted_texts": error_texts}


async def _summarize_reports_node(state: UnifiedGraphState, summarize_chain) -> UnifiedGraphState:
    """Node wrapper for summarizing extracted text using research_tools."""
    print("--- Node: summarize_reports (using research_tools) ---")
    if state.get("error_message"):
        logger.warning("Skipping summarize_reports due to previous error.")
        return {}
    extracted_texts = state.get("extracted_texts")
    if not extracted_texts:
        logger.warning("No extracted texts found in state for summarize_reports.")
        return {"individual_summaries": {}}
    try:
        individual_summaries = await summarize_extracted_texts(extracted_texts, summarize_chain)
        logger.info(f"Summarize node finished. Returning summaries: {list(individual_summaries.keys())}")
        return {"individual_summaries": individual_summaries}
    except Exception as e:
        logger.error(f"Error calling summarize_extracted_texts: {e}", exc_info=True)
        error_summaries = {company: f"Error in summarization step: {e}" for company in extracted_texts}
        return {
            "individual_summaries": error_summaries,
            "error_message": f"Failed during summarization: {e}",
        }


def _synthesize_trends_node(state: UnifiedGraphState, llm, synthesis_prompt_template_str: str) -> UnifiedGraphState:
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
        logger.warning("No valid summaries available for synthesis.")
        return {"error_message": "Analysis failed: No valid summaries could be generated."}
    logger.info("Synthesizing trends across summaries.")
    combined_summaries = "\n\n".join(
        [f"--- Summary for {company} ---\n{summary}" for company, summary in valid_summaries.items()]
    )
    try:
        synthesis_prompt = synthesis_prompt_template_str.format(
            industry=industry, combined_summaries=combined_summaries
        )
        final_synthesis_message = llm.invoke(synthesis_prompt)
        final_synthesis = final_synthesis_message.content
        report_list = "\n".join(
            [f"- {comp}: {report_urls.get(comp, 'URL not found/processed')}" for comp in valid_summaries.keys()]
        )
        if not report_list:
            report_list = "No report URLs were successfully processed."
        result_header = "Analysis based on reports processed for:\n"
        report_section = f"{report_list}\n\n"
        trends_header = "--- Synthesized Trends ---\n"
        final_result = result_header + report_section + trends_header + final_synthesis
        logger.info("Finished synthesizing trends.")
        return {"synthesis_result": final_result}
    except Exception as e:
        logger.error(f"Error during final synthesis: {e}", exc_info=True)
        return {"error_message": f"Failed during the final synthesis step: {e}"}


def _handle_error_node(state: UnifiedGraphState) -> UnifiedGraphState:
    """Handles errors captured in the state."""
    print("--- Node: handle_error ---")
    error = state.get("error_message", "Unknown error")
    logger.error(f"Graph execution failed: {error}")
    return {"error_message": error}


def _route_research_step_conditional_edge(state: UnifiedGraphState) -> str:
    """Checks for errors and decides whether to continue or handle error."""
    logger.debug(f"Routing research step. Current state keys: {state.keys()}")
    logger.debug(f"Checking for error_message: {state.get('error_message')}")
    if state.get("error_message"):
        logger.warning("Error detected in research step, routing to handle_error.")
        return ROUTE_HANDLE_ERROR_STEP
    else:
        logger.info("No error detected in research step, continuing.")
        return ROUTE_CONTINUE


# --- Build the Unified Graph ---


def build_unified_graph():
    """Builds the unified LangGraph workflow, initializing components inside."""

    try:
        llm, llm_summarizer = initialize_gemini()
        react_prompt_template = load_prompt()

        with open(MAP_PROMPT_FILE, "r") as f:
            map_prompt_template = f.read()
        map_prompt = PromptTemplate.from_template(map_prompt_template)
        with open(COMBINE_PROMPT_FILE, "r") as f:
            combine_prompt_template = f.read()
        combine_prompt = PromptTemplate.from_template(combine_prompt_template)

        with open(COMPANY_ID_PROMPT_FILE, "r") as f:
            company_prompt_template_str = f.read()
        with open(SYNTHESIS_PROMPT_FILE, "r") as f:
            synthesis_prompt_template_str = f.read()

        is_debug_enabled = logging.getLogger().getEffectiveLevel() <= logging.DEBUG
        summarize_chain = load_summarize_chain(
            llm=llm_summarizer,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=is_debug_enabled,
        )
        logger.info(f"Graph Builder: Summarize chain initialized (verbose={is_debug_enabled}).")

        react_agent = create_react_agent(llm, BASIC_AGENT_TOOLS, react_prompt_template)
        logger.info("Graph Builder: ReAct agent initialized.")

    except Exception as e:
        logger.error(f"Graph Builder: Failed to initialize LLMs or chains: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize graph components: {e}") from e

    workflow = StateGraph(UnifiedGraphState)

    workflow.add_node("route_request", route_request)
    workflow.add_node(
        "run_basic_agent",
        functools.partial(_run_basic_agent_node, react_agent=react_agent, basic_agent_tools=BASIC_AGENT_TOOLS),
    )
    workflow.add_node(
        "identify_companies",
        functools.partial(_identify_companies_node, llm=llm, company_prompt_template_str=company_prompt_template_str),
    )
    workflow.add_node("search_for_reports", _search_for_reports_node)
    workflow.add_node("download_and_extract", _download_and_extract_node)
    workflow.add_node("summarize_reports", functools.partial(_summarize_reports_node, summarize_chain=summarize_chain))
    workflow.add_node(
        "synthesize_trends",
        functools.partial(
            _synthesize_trends_node, llm=llm, synthesis_prompt_template_str=synthesis_prompt_template_str
        ),
    )
    workflow.add_node("handle_error", _handle_error_node)

    workflow.set_entry_point("route_request")

    workflow.add_conditional_edges(
        "route_request",
        decide_route,
        {
            ROUTE_BASIC_AGENT_PATH: "run_basic_agent",
            ROUTE_RESEARCH_PATH: "identify_companies",
            ROUTE_ERROR_INPUT: "handle_error",
        },
    )

    workflow.add_conditional_edges(
        "identify_companies",
        _route_research_step_conditional_edge,
        {ROUTE_CONTINUE: "search_for_reports", ROUTE_HANDLE_ERROR_STEP: "handle_error"},
    )
    workflow.add_conditional_edges(
        "search_for_reports",
        _route_research_step_conditional_edge,
        {ROUTE_CONTINUE: "download_and_extract", ROUTE_HANDLE_ERROR_STEP: "handle_error"},
    )
    workflow.add_conditional_edges(
        "download_and_extract",
        _route_research_step_conditional_edge,
        {ROUTE_CONTINUE: "summarize_reports", ROUTE_HANDLE_ERROR_STEP: "handle_error"},
    )
    workflow.add_conditional_edges(
        "summarize_reports",
        _route_research_step_conditional_edge,
        {ROUTE_CONTINUE: "synthesize_trends", ROUTE_HANDLE_ERROR_STEP: "handle_error"},
    )
    workflow.add_conditional_edges(
        "synthesize_trends",
        _route_research_step_conditional_edge,
        {ROUTE_CONTINUE: END, ROUTE_HANDLE_ERROR_STEP: "handle_error"},
    )

    workflow.add_edge("run_basic_agent", END)
    workflow.add_edge("handle_error", END)

    app = workflow.compile()
    logging.info("Unified LangGraph workflow compiled.")
    return app
