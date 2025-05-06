import asyncio
import logging
import os

import langchain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from logging_config import setup_logging

# Project-specific imports
from research_agent.agent import initialize_gemini
from research_agent.file_tools import download_pdf_tool, extract_pdf_text_tool
from research_agent.research_tools import _process_single_summary  # Reusing this function

setup_logging()
logger = logging.getLogger(__name__)

# Disable LangChain's global verbosity to suppress operational logs
langchain.verbose = False
logger.info("LangChain global verbosity disabled.")


def get_summarize_chain():
    """Initializes and returns the summarization chain."""
    try:
        # We only need the summarizer LLM from the tuple
        _, llm_summarizer = initialize_gemini()

        # Determine prompts directory relative to this file's location
        # Assuming single_pdf_analyzer.py is in sustainability_research_agent/
        # and prompts are in sustainability_research_agent/prompts/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, "prompts")

        if not os.path.isdir(prompts_dir):
            # Fallback if running from project root and current_dir is not as expected
            project_root_prompts_dir = os.path.join(os.getcwd(), "sustainability_research_agent", "prompts")
            if os.path.isdir(project_root_prompts_dir):
                prompts_dir = project_root_prompts_dir
            else:
                logger.error(f"Prompts directory not found at {prompts_dir} or {project_root_prompts_dir}")
                raise FileNotFoundError(
                    f"Prompts directory not found. Looked in {prompts_dir} and {project_root_prompts_dir}"
                )

        with open(os.path.join(prompts_dir, "map_prompt.txt"), "r") as f:
            map_prompt_template = f.read()
        map_prompt = PromptTemplate.from_template(map_prompt_template)

        with open(os.path.join(prompts_dir, "combine_prompt.txt"), "r") as f:
            combine_prompt_template = f.read()
        combine_prompt = PromptTemplate.from_template(combine_prompt_template)

        summarize_chain = load_summarize_chain(
            llm=llm_summarizer,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False,  # Explicitly disable LangChain's verbose operational logs for this chain
            return_intermediate_steps=True,  # To get map stage outputs for debugging
            # Pass token_max to the ReduceDocumentsChain to handle larger combined documents in fewer steps
            token_max=250000,  # Pass token_max to the ReduceDocumentsChain
        )
        logger.info(
            "Single PDF Analyzer: Summarize chain initialized (chain verbosity disabled, "
            "returning intermediate steps, increased reduce token_max)."
        )
        return summarize_chain
    except Exception as e:
        logger.error(f"Failed to initialize summarization chain: {e}", exc_info=True)
        raise


def extract_text_from_pdf(pdf_path_or_url: str) -> str:
    """Downloads (if URL) and extracts text from a PDF."""
    local_pdf_path = pdf_path_or_url
    if pdf_path_or_url.startswith("http://") or pdf_path_or_url.startswith("https://"):
        logger.info(f"Downloading PDF from URL: {pdf_path_or_url}")
        # download_pdf_tool from file_tools handles its own caching for downloads
        local_pdf_path = download_pdf_tool.run(pdf_path_or_url)
        if local_pdf_path.startswith("Error"):
            logger.error(f"Failed to download PDF: {local_pdf_path}")
            return local_pdf_path  # Return error message
    else:
        if not os.path.exists(local_pdf_path):
            logger.error(f"Local PDF file not found: {local_pdf_path}")
            return f"Error: Local PDF file not found: {local_pdf_path}"

    logger.info(f"Extracting text from PDF: {local_pdf_path}")
    extracted_text = extract_pdf_text_tool.run(
        local_pdf_path
    )  # extract_pdf_text_tool handles its own caching for extractions

    # Log the beginning of the extracted text for diagnostics
    if extracted_text:
        logger.debug(f"Beginning of extracted text from {local_pdf_path} (first 500 chars): {extracted_text[:500]}")
    else:
        logger.debug(f"No text extracted from {local_pdf_path}.")

    if extracted_text.startswith("Error") or extracted_text.startswith("Warning") or not extracted_text.strip():
        logger.warning(
            f"Failed to extract text or text is empty from {local_pdf_path}: {extracted_text[:200]}..."
        )  # Log only start of message
        return extracted_text  # Return error/warning or empty text message

    logger.info(f"Successfully extracted text (length: {len(extracted_text)}) from {local_pdf_path}.")
    return extracted_text


async def main():
    pdf_input = "pdf_cache/2024_Sustainability_En_b76ea15980.pdf"
    # Use filename or last part of URL as identifier for logging/caching within _process_single_summary
    pdf_identifier = os.path.basename(pdf_input)

    logger.info(f"--- Starting Single PDF Analysis for: {pdf_input} (Identifier: {pdf_identifier}) ---")

    try:
        summarize_chain = get_summarize_chain()
    except Exception as e:
        logger.critical(f"Could not initialize summarization chain. Exiting. Error: {e}", exc_info=True)
        print(f"FATAL: Could not initialize components. Check logs. Error: {e}")
        return

    extracted_text = extract_text_from_pdf(pdf_input)

    if extracted_text.startswith("Error") or extracted_text.startswith("Warning") or not extracted_text.strip():
        print(f"\n--- Analysis Failed for {pdf_identifier} ---")
        print(f"Reason: {extracted_text}")
        logger.error(f"Analysis failed for {pdf_identifier} due to text extraction issue: {extracted_text}")
        return

    MIN_TEXT_LENGTH = 200
    if len(extracted_text) < MIN_TEXT_LENGTH:
        message = (
            f"Extracted text is too short (length: {len(extracted_text)}, "
            f"min: {MIN_TEXT_LENGTH}) to be meaningfully summarized."
        )
        print(f"\n--- Analysis Skipped for {pdf_identifier} ---")
        print(f"Reason: {message}")
        logger.warning(f"Analysis skipped for {pdf_identifier}: {message}")
        return

    logger.info(f"Calling _process_single_summary for {pdf_identifier}")
    _, summary_result = await _process_single_summary(pdf_identifier, extracted_text, summarize_chain)

    print(f"\n--- Summary for {pdf_identifier} ---")
    print(summary_result)
    logger.info(f"--- Finished Single PDF Analysis for: {pdf_input} ---")


if __name__ == "__main__":
    asyncio.run(main())
