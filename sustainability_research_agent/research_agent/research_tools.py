import logging
import os
from typing import Dict

# Removed: from research_agent.agent import llm_summarizer # LLM/Chain initialized elsewhere
from research_agent.file_tools import (
    download_pdf_tool,
    extract_pdf_text_tool,
)  # Import relative to sustainability_research_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Define Cache Directory ---
# Ensure this path is correct relative to where this module might be run from
# Or consider making it configurable/absolute
SUMMARY_CACHE_DIR = os.path.join(os.path.dirname(__file__), "summary_cache")
os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)
logging.info(f"Research Tools: Using summary cache directory: {SUMMARY_CACHE_DIR}")

# --- Initialize Components ---

# Setup for text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
logging.info("Research Tools: Initialized text splitter.")

# Removed module-level summarize_chain initialization. It will be passed into functions.


def download_and_extract_reports(report_urls: Dict[str, str]) -> Dict[str, str]:
    """Downloads PDFs from URLs and extracts text.

    Args:
        report_urls: A dictionary mapping company names to report PDF URLs.

    Returns:
        A dictionary mapping company names to extracted text or error messages.
    """
    extracted_texts: Dict[str, str] = {}
    logging.info("Research Tools: Downloading and extracting text from found PDFs.")

    for company, url in report_urls.items():
        if url is None or url.startswith("Error"):
            logging.warning(
                f"Research Tools: Skipping download/extraction for {company} due to missing/error URL: {url}"
            )
            extracted_texts[company] = "No valid URL found."
            continue

        logging.info(f"Research Tools: Processing PDF for {company} from URL: {url}")
        try:
            # Download (uses cache via file_tools)
            local_path = download_pdf_tool.run(url)
            if local_path.startswith("Error"):
                logging.error(
                    f"Research Tools: Download failed for {company}: {local_path}"
                )
                extracted_texts[company] = f"Download failed: {local_path}"
                continue

            # Extract Text
            extracted_text = extract_pdf_text_tool.run(local_path)
            if extracted_text.startswith("Error") or extracted_text.startswith(
                "Warning"
            ):
                logging.warning(
                    f"Research Tools: Text extraction failed/empty for {company}: {extracted_text}"
                )
                extracted_texts[company] = extracted_text  # Store warning/error
            else:
                logging.info(
                    f"Research Tools: Successfully extracted text for {company} (length: {len(extracted_text)})."
                )
                extracted_texts[company] = extracted_text
        except Exception as e:
            logging.error(
                f"Research Tools: Error during download/extraction for {company}: {e}",
                exc_info=True,
            )
            extracted_texts[company] = f"Error during download/extraction: {e}"

    return extracted_texts


def summarize_extracted_texts(
    extracted_texts: Dict[str, str], summarize_chain
) -> Dict[str, str]:
    """Summarizes the extracted text for each report, using a cache.

    Args:
        extracted_texts: A dictionary mapping company names to extracted text.
        summarize_chain: The pre-configured LangChain summarization chain instance.

    Returns:
        A dictionary mapping company names to summaries or error messages.
    """
    if summarize_chain is None:
        logging.error(
            "Research Tools: Summarize chain was not provided. Cannot summarize."
        )
        # Return errors for all companies
        return {
            company: "Error: Summarization chain not provided."
            for company in extracted_texts
        }

    individual_summaries: Dict[str, str] = {}
    logging.info("Research Tools: Summarizing extracted report texts.")

    for company, text in extracted_texts.items():
        if (
            text is None
            or text.startswith("Error")
            or text.startswith("Warning")
            or text == "No valid URL found."
            or not text.strip()
        ):
            logging.warning(
                f"Research Tools: Skipping summarization for {company} due to previous error or empty text."
            )
            individual_summaries[company] = (
                "Skipped due to previous error or empty text."
            )
            continue

        # Use a simple filename based on company name for cache.
        cache_filename = f"{company.replace(' ', '_').lower()}_summary.txt"
        cache_filepath = os.path.join(SUMMARY_CACHE_DIR, cache_filename)

        if os.path.exists(cache_filepath):
            logging.info(
                f"Research Tools: Cache hit for {company} summary: Loading from {cache_filepath}"
            )
            try:
                with open(cache_filepath, "r") as f:
                    summary = f.read()
                individual_summaries[company] = summary
                continue  # Move to next company
            except Exception as e:
                logging.warning(
                    f"Research Tools: Failed to read cached summary for {company} from {cache_filepath}: {e}. Will regenerate."
                )

        # If cache miss or read error, generate summary
        logging.info(f"Research Tools: Cache miss for {company}. Generating summary...")
        try:
            docs = text_splitter.create_documents([text])
            summary = summarize_chain.run(docs)  # Uses llm_summarizer with timeout
            individual_summaries[company] = summary
            logging.info(f"Research Tools: Finished generating summary for {company}.")

            # Save summary to cache
            try:
                with open(cache_filepath, "w") as f:
                    f.write(summary)
                logging.info(
                    f"Research Tools: Saved summary for {company} to cache: {cache_filepath}"
                )
            except Exception as e:
                logging.error(
                    f"Research Tools: Failed to save summary to cache for {company} at {cache_filepath}: {e}"
                )

        except Exception as e:
            logging.error(
                f"Research Tools: Error summarizing report for {company}: {e}",
                exc_info=True,
            )
            individual_summaries[company] = f"Error during summarization: {e}"

    return individual_summaries
